from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class IPTun(object):

    def __init__(self, module):
        self.module = module
        self.name = module.params['name']
        self.type = module.params['type']
        self.local_address = module.params['local_address']
        self.remote_address = module.params['remote_address']
        self.temporary = module.params['temporary']
        self.state = module.params['state']
        self.dladm_bin = self.module.get_bin_path('dladm', True)

    def iptun_exists(self):
        cmd = [self.dladm_bin]
        cmd.append('show-iptun')
        cmd.append(self.name)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def create_iptun(self):
        cmd = [self.dladm_bin]
        cmd.append('create-iptun')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-T')
        cmd.append(self.type)
        cmd.append('-a')
        cmd.append('local=' + self.local_address + ',remote=' + self.remote_address)
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def delete_iptun(self):
        cmd = [self.dladm_bin]
        cmd.append('delete-iptun')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def update_iptun(self):
        cmd = [self.dladm_bin]
        cmd.append('modify-iptun')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-a')
        cmd.append('local=' + self.local_address + ',remote=' + self.remote_address)
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def _query_iptun_props(self):
        cmd = [self.dladm_bin]
        cmd.append('show-iptun')
        cmd.append('-p')
        cmd.append('-c')
        cmd.append('link,type,flags,local,remote')
        cmd.append(self.name)
        return self.module.run_command(cmd)

    def iptun_needs_updating(self):
        rc, out, err = self._query_iptun_props()
        NEEDS_UPDATING = False
        if rc == 0:
            configured_local, configured_remote = out.split(':')[3:]
            if self.local_address != configured_local or self.remote_address != configured_remote:
                NEEDS_UPDATING = True
            return NEEDS_UPDATING
        else:
            self.module.fail_json(msg='Failed to query tunnel interface %s properties' % self.name, err=err, rc=rc)