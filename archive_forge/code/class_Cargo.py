from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
class Cargo(object):

    def __init__(self, module, **kwargs):
        self.module = module
        self.executable = [kwargs['executable'] or module.get_bin_path('cargo', True)]
        self.name = kwargs['name']
        self.path = kwargs['path']
        self.state = kwargs['state']
        self.version = kwargs['version']
        self.locked = kwargs['locked']

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if path is not None and (not os.path.isdir(path)):
            self.module.fail_json(msg='Path %s is not a directory' % path)
        self._path = path

    def _exec(self, args, run_in_check_mode=False, check_rc=True, add_package_name=True):
        if not self.module.check_mode or (self.module.check_mode and run_in_check_mode):
            cmd = self.executable + args
            rc, out, err = self.module.run_command(cmd, check_rc=check_rc)
            return (out, err)
        return ('', '')

    def get_installed(self):
        cmd = ['install', '--list']
        if self.path:
            cmd.append('--root')
            cmd.append(self.path)
        data, dummy = self._exec(cmd, True, False, False)
        package_regex = re.compile('^([\\w\\-]+) v(.+):$')
        installed = {}
        for line in data.splitlines():
            package_info = package_regex.match(line)
            if package_info:
                installed[package_info.group(1)] = package_info.group(2)
        return installed

    def install(self, packages=None):
        cmd = ['install']
        cmd.extend(packages or self.name)
        if self.locked:
            cmd.append('--locked')
        if self.path:
            cmd.append('--root')
            cmd.append(self.path)
        if self.version:
            cmd.append('--version')
            cmd.append(self.version)
        return self._exec(cmd)

    def is_outdated(self, name):
        installed_version = self.get_installed().get(name)
        cmd = ['search', name, '--limit', '1']
        data, dummy = self._exec(cmd, True, False, False)
        match = re.search('"(.+)"', data)
        if match:
            latest_version = match.group(1)
        return installed_version != latest_version

    def uninstall(self, packages=None):
        cmd = ['uninstall']
        cmd.extend(packages or self.name)
        return self._exec(cmd)