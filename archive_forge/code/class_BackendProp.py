from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class BackendProp(object):

    def __init__(self, module):
        self._module = module

    def get_property(self, opendj_bindir, hostname, port, username, password_method, backend_name):
        my_command = [opendj_bindir + '/dsconfig', 'get-backend-prop', '-h', hostname, '--port', str(port), '--bindDN', username, '--backend-name', backend_name, '-n', '-X', '-s'] + password_method
        rc, stdout, stderr = self._module.run_command(my_command)
        if rc == 0:
            return stdout
        else:
            self._module.fail_json(msg='Error message: ' + str(stderr))

    def set_property(self, opendj_bindir, hostname, port, username, password_method, backend_name, name, value):
        my_command = [opendj_bindir + '/dsconfig', 'set-backend-prop', '-h', hostname, '--port', str(port), '--bindDN', username, '--backend-name', backend_name, '--set', name + ':' + value, '-n', '-X'] + password_method
        rc, stdout, stderr = self._module.run_command(my_command)
        if rc == 0:
            return True
        else:
            self._module.fail_json(msg='Error message: ' + stderr)

    def validate_data(self, data=None, name=None, value=None):
        for config_line in data.split('\n'):
            if config_line:
                split_line = config_line.split()
                if split_line[0] == name:
                    if split_line[1] == value:
                        return True
        return False