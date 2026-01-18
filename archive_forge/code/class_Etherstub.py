from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class Etherstub(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def etherstub_exists(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('show-etherstub')
        cmd.append(self.name)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def create_etherstub(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('create-etherstub')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def delete_etherstub(self):
        cmd = [self.module.get_bin_path('dladm', True)]
        cmd.append('delete-etherstub')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)