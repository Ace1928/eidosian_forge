from __future__ import absolute_import, division, print_function
import socket
from ansible.module_utils.basic import AnsibleModule
class Addr(object):

    def __init__(self, module):
        self.module = module
        self.address = module.params['address']
        self.addrtype = module.params['addrtype']
        self.addrobj = module.params['addrobj']
        self.temporary = module.params['temporary']
        self.state = module.params['state']
        self.wait = module.params['wait']

    def is_cidr_notation(self):
        return self.address.count('/') == 1

    def is_valid_address(self):
        ip_address = self.address.split('/')[0]
        try:
            if len(ip_address.split('.')) == 4:
                socket.inet_pton(socket.AF_INET, ip_address)
            else:
                socket.inet_pton(socket.AF_INET6, ip_address)
        except socket.error:
            return False
        return True

    def is_dhcp(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-addr')
        cmd.append('-p')
        cmd.append('-o')
        cmd.append('type')
        cmd.append(self.addrobj)
        rc, out, err = self.module.run_command(cmd)
        if rc == 0:
            if out.rstrip() != 'dhcp':
                return False
            return True
        else:
            self.module.fail_json(msg='Wrong addrtype %s for addrobj "%s": %s' % (out, self.addrobj, err), rc=rc, stderr=err)

    def addrobj_exists(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-addr')
        cmd.append(self.addrobj)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            return False

    def delete_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('delete-addr')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def create_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('create-addr')
        cmd.append('-T')
        cmd.append(self.addrtype)
        if self.temporary:
            cmd.append('-t')
        if self.addrtype == 'static':
            cmd.append('-a')
            cmd.append(self.address)
        if self.addrtype == 'dhcp' and self.wait:
            cmd.append('-w')
            cmd.append(self.wait)
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def up_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('up-addr')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def down_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('down-addr')
        if self.temporary:
            cmd.append('-t')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def enable_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('enable-addr')
        cmd.append('-t')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def disable_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('disable-addr')
        cmd.append('-t')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def refresh_addr(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('refresh-addr')
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)