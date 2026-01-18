from __future__ import (absolute_import, division, print_function)
import base64
import errno
import json
import os
import pkgutil
import random
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.compat.importlib import import_module
from ansible.plugins.loader import ps_module_utils_loader
from ansible.utils.collection_loader import resource_from_fqcr
class PSModuleDepFinder(object):

    def __init__(self):
        self.ps_modules = dict()
        self.exec_scripts = dict()
        self.cs_utils_wrapper = dict()
        self.cs_utils_module = dict()
        self.ps_version = None
        self.os_version = None
        self.become = False
        self._re_cs_module = [re.compile(to_bytes('(?i)^using\\s((Ansible\\..+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+));\\s*$'))]
        self._re_cs_in_ps_module = [re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-csharputil\\s+((Ansible\\.[\\w\\.]+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+)|(\\.[\\w\\.]+))(?P<optional>\\s+-Optional){0,1}'))]
        self._re_ps_module = [re.compile(to_bytes('(?i)^#\\s*requires\\s+\\-module(?:s?)\\s*(Ansible\\.ModuleUtils\\..+)')), re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-powershell\\s+((Ansible\\.ModuleUtils\\.[\\w\\.]+)|(ansible_collections\\.\\w+\\.\\w+\\.plugins\\.module_utils\\.[\\w\\.]+)|(\\.[\\w\\.]+))(?P<optional>\\s+-Optional){0,1}'))]
        self._re_wrapper = re.compile(to_bytes('(?i)^#\\s*ansiblerequires\\s+-wrapper\\s+(\\w*)'))
        self._re_ps_version = re.compile(to_bytes('(?i)^#requires\\s+\\-version\\s+([0-9]+(\\.[0-9]+){0,3})$'))
        self._re_os_version = re.compile(to_bytes('(?i)^#ansiblerequires\\s+\\-osversion\\s+([0-9]+(\\.[0-9]+){0,3})$'))
        self._re_become = re.compile(to_bytes('(?i)^#ansiblerequires\\s+\\-become$'))

    def scan_module(self, module_data, fqn=None, wrapper=False, powershell=True):
        lines = module_data.split(b'\n')
        module_utils = set()
        if wrapper:
            cs_utils = self.cs_utils_wrapper
        else:
            cs_utils = self.cs_utils_module
        if powershell:
            checks = [(self._re_ps_module, self.ps_modules, '.psm1'), (self._re_cs_in_ps_module, cs_utils, '.cs')]
        else:
            checks = [(self._re_cs_module, cs_utils, '.cs')]
        for line in lines:
            for check in checks:
                for pattern in check[0]:
                    match = pattern.match(line)
                    if match:
                        module_util_name = to_text(match.group(1).rstrip())
                        match_dict = match.groupdict()
                        optional = match_dict.get('optional', None) is not None
                        if module_util_name not in check[1].keys():
                            module_utils.add((module_util_name, check[2], fqn, optional))
                        break
            if powershell:
                ps_version_match = self._re_ps_version.match(line)
                if ps_version_match:
                    self._parse_version_match(ps_version_match, 'ps_version')
                os_version_match = self._re_os_version.match(line)
                if os_version_match:
                    self._parse_version_match(os_version_match, 'os_version')
                if not self.become:
                    become_match = self._re_become.match(line)
                    if become_match:
                        self.become = True
            if wrapper:
                wrapper_match = self._re_wrapper.match(line)
                if wrapper_match:
                    self.scan_exec_script(wrapper_match.group(1).rstrip())
        for m in set(module_utils):
            self._add_module(*m, wrapper=wrapper)

    def scan_exec_script(self, name):
        name = to_text(name)
        if name in self.exec_scripts.keys():
            return
        data = pkgutil.get_data('ansible.executor.powershell', to_native(name + '.ps1'))
        if data is None:
            raise AnsibleError("Could not find executor powershell script for '%s'" % name)
        b_data = to_bytes(data)
        if C.DEFAULT_DEBUG:
            exec_script = b_data
        else:
            exec_script = _strip_comments(b_data)
        self.exec_scripts[name] = to_bytes(exec_script)
        self.scan_module(b_data, wrapper=True, powershell=True)

    def _add_module(self, name, ext, fqn, optional, wrapper=False):
        m = to_text(name)
        util_fqn = None
        if m.startswith('Ansible.'):
            mu_path = ps_module_utils_loader.find_plugin(m, ext)
            if not mu_path:
                if optional:
                    return
                raise AnsibleError("Could not find imported module support code for '%s'" % m)
            module_util_data = to_bytes(_slurp(mu_path))
        else:
            submodules = m.split('.')
            if m.startswith('.'):
                fqn_submodules = fqn.split('.')
                for submodule in submodules:
                    if submodule:
                        break
                    del fqn_submodules[-1]
                submodules = fqn_submodules + [s for s in submodules if s]
            n_package_name = to_native('.'.join(submodules[:-1]), errors='surrogate_or_strict')
            n_resource_name = to_native(submodules[-1] + ext, errors='surrogate_or_strict')
            try:
                module_util = import_module(n_package_name)
                pkg_data = pkgutil.get_data(n_package_name, n_resource_name)
                if pkg_data is None:
                    raise ImportError('No package data found')
                module_util_data = to_bytes(pkg_data, errors='surrogate_or_strict')
                util_fqn = to_text('%s.%s ' % (n_package_name, submodules[-1]), errors='surrogate_or_strict')
                resource_paths = list(module_util.__path__)
                if len(resource_paths) != 1:
                    raise AnsibleError("Internal error: Referenced module_util package '%s' contains 0 or multiple import locations when we only expect 1." % n_package_name)
                mu_path = os.path.join(resource_paths[0], n_resource_name)
            except (ImportError, OSError) as err:
                if getattr(err, 'errno', errno.ENOENT) == errno.ENOENT:
                    if optional:
                        return
                    raise AnsibleError("Could not find collection imported module support code for '%s'" % to_native(m))
                else:
                    raise
        util_info = {'data': module_util_data, 'path': to_text(mu_path)}
        if ext == '.psm1':
            self.ps_modules[m] = util_info
        elif wrapper:
            self.cs_utils_wrapper[m] = util_info
        else:
            self.cs_utils_module[m] = util_info
        self.scan_module(module_util_data, fqn=util_fqn, wrapper=wrapper, powershell=ext == '.psm1')

    def _parse_version_match(self, match, attribute):
        new_version = to_text(match.group(1)).rstrip()
        if match.group(2) is None:
            new_version = '%s.0' % new_version
        existing_version = getattr(self, attribute, None)
        if existing_version is None:
            setattr(self, attribute, new_version)
        elif LooseVersion(new_version) > LooseVersion(existing_version):
            setattr(self, attribute, new_version)