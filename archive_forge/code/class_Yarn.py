from __future__ import absolute_import, division, print_function
import os
import json
from ansible.module_utils.basic import AnsibleModule
class Yarn(object):

    def __init__(self, module, **kwargs):
        self.module = module
        self.globally = kwargs['globally']
        self.name = kwargs['name']
        self.version = kwargs['version']
        self.path = kwargs['path']
        self.registry = kwargs['registry']
        self.production = kwargs['production']
        self.ignore_scripts = kwargs['ignore_scripts']
        self.executable = kwargs['executable']
        self.name_version = None
        if kwargs['version'] and self.name is not None:
            self.name_version = self.name + '@' + str(self.version)
        elif self.name is not None:
            self.name_version = self.name

    def _exec(self, args, run_in_check_mode=False, check_rc=True, unsupported_with_global=False):
        if not self.module.check_mode or (self.module.check_mode and run_in_check_mode):
            with_global_arg = self.globally and (not unsupported_with_global)
            if with_global_arg:
                args.insert(0, 'global')
            cmd = self.executable + args
            if self.production:
                cmd.append('--production')
            if self.ignore_scripts:
                cmd.append('--ignore-scripts')
            if self.registry:
                cmd.append('--registry')
                cmd.append(self.registry)
            cwd = None
            if self.path and (not with_global_arg):
                if not os.path.exists(self.path):
                    os.makedirs(self.path)
                if not os.path.isdir(self.path):
                    self.module.fail_json(msg='Path provided %s is not a directory' % self.path)
                cwd = self.path
                if not os.path.isfile(os.path.join(self.path, 'package.json')):
                    self.module.fail_json(msg='Package.json does not exist in provided path.')
            rc, out, err = self.module.run_command(cmd, check_rc=check_rc, cwd=cwd)
            return (out, err)
        return (None, None)

    def _process_yarn_error(self, err):
        try:
            for line in err.splitlines():
                if json.loads(line)['type'] == 'error':
                    self.module.fail_json(msg=err)
        except Exception:
            self.module.fail_json(msg='Unexpected stderr output from Yarn: %s' % err, stderr=err)

    def list(self):
        cmd = ['list', '--depth=0', '--json']
        installed = list()
        missing = list()
        if not os.path.isfile(os.path.join(self.path, 'yarn.lock')):
            missing.append(self.name)
            return (installed, missing)
        result, error = self._exec(cmd, run_in_check_mode=True, check_rc=False, unsupported_with_global=True)
        self._process_yarn_error(error)
        for json_line in result.strip().split('\n'):
            data = json.loads(json_line)
            if data['type'] == 'tree':
                dependencies = data['data']['trees']
                for dep in dependencies:
                    name, version = dep['name'].rsplit('@', 1)
                    installed.append(name)
        if self.name not in installed:
            missing.append(self.name)
        return (installed, missing)

    def install(self):
        if self.name_version:
            return self._exec(['add', self.name_version])
        return self._exec(['install', '--non-interactive'])

    def update(self):
        return self._exec(['upgrade', '--latest'])

    def uninstall(self):
        return self._exec(['remove', self.name])

    def list_outdated(self):
        outdated = list()
        if not os.path.isfile(os.path.join(self.path, 'yarn.lock')):
            return outdated
        cmd_result, err = self._exec(['outdated', '--json'], True, False, unsupported_with_global=True)
        self._process_yarn_error(err)
        if not cmd_result:
            return outdated
        outdated_packages_data = cmd_result.splitlines()[1]
        data = json.loads(outdated_packages_data)
        try:
            outdated_dependencies = data['data']['body']
        except KeyError:
            return outdated
        for dep in outdated_dependencies:
            outdated.append(dep[0])
        return outdated