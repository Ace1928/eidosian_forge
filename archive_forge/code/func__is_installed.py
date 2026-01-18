from __future__ import absolute_import, division, print_function
import os
import re
import sys
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_file
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.common.respawn import has_respawned, probe_interpreters_for_module, respawn_module
from ansible.module_utils.yumdnf import YumDnf, yumdnf_argument_spec
def _is_installed(self, pkg):
    installed = self.base.sack.query().installed()
    package_spec = {}
    name, arch = self._split_package_arch(pkg)
    if arch:
        package_spec['arch'] = arch
    package_details = self._packagename_dict(pkg)
    if package_details:
        package_details['epoch'] = int(package_details['epoch'])
        package_spec.update(package_details)
    else:
        package_spec['name'] = name
    return bool(installed.filter(**package_spec))