from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
class Sudoers(object):
    FILE_MODE = 288

    def __init__(self, module):
        self.module = module
        self.check_mode = module.check_mode
        self.name = module.params['name']
        self.user = module.params['user']
        self.group = module.params['group']
        self.state = module.params['state']
        self.noexec = module.params['noexec']
        self.nopassword = module.params['nopassword']
        self.setenv = module.params['setenv']
        self.host = module.params['host']
        self.runas = module.params['runas']
        self.sudoers_path = module.params['sudoers_path']
        self.file = os.path.join(self.sudoers_path, self.name)
        self.commands = module.params['commands']
        self.validation = module.params['validation']

    def write(self):
        if self.check_mode:
            return
        with open(self.file, 'w') as f:
            f.write(self.content())
        os.chmod(self.file, self.FILE_MODE)

    def delete(self):
        if self.check_mode:
            return
        os.remove(self.file)

    def exists(self):
        return os.path.exists(self.file)

    def matches(self):
        with open(self.file, 'r') as f:
            content_matches = f.read() == self.content()
        current_mode = os.stat(self.file).st_mode & 511
        mode_matches = current_mode == self.FILE_MODE
        return content_matches and mode_matches

    def content(self):
        if self.user:
            owner = self.user
        elif self.group:
            owner = '%{group}'.format(group=self.group)
        commands_str = ', '.join(self.commands)
        noexec_str = 'NOEXEC:' if self.noexec else ''
        nopasswd_str = 'NOPASSWD:' if self.nopassword else ''
        setenv_str = 'SETENV:' if self.setenv else ''
        runas_str = '({runas})'.format(runas=self.runas) if self.runas is not None else ''
        return '{owner} {host}={runas}{noexec}{nopasswd}{setenv} {commands}\n'.format(owner=owner, host=self.host, runas=runas_str, noexec=noexec_str, nopasswd=nopasswd_str, setenv=setenv_str, commands=commands_str)

    def validate(self):
        if self.validation == 'absent':
            return
        visudo_path = self.module.get_bin_path('visudo', required=self.validation == 'required')
        if visudo_path is None:
            return
        check_command = [visudo_path, '-c', '-f', '-']
        rc, stdout, stderr = self.module.run_command(check_command, data=self.content())
        if rc != 0:
            raise Exception('Failed to validate sudoers rule:\n{stdout}'.format(stdout=stdout))

    def run(self):
        if self.state == 'absent':
            if self.exists():
                self.delete()
                return True
            else:
                return False
        self.validate()
        if self.exists() and self.matches():
            return False
        self.write()
        return True