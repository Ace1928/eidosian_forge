from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
class AddrProp(object):

    def __init__(self, module):
        self.module = module
        self.addrobj = module.params['addrobj']
        self.property = module.params['property']
        self.value = module.params['value']
        self.temporary = module.params['temporary']
        self.state = module.params['state']

    def property_exists(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-addrprop')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.addrobj)
        rc, dummy, dummy = self.module.run_command(cmd)
        if rc == 0:
            return True
        else:
            self.module.fail_json(msg='Unknown property "%s" on addrobj %s' % (self.property, self.addrobj), property=self.property, addrobj=self.addrobj)

    def property_is_modified(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-addrprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('current,default')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.addrobj)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        value, default = out.split(':')
        if rc == 0 and value == default:
            return True
        else:
            return False

    def property_is_set(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('show-addrprop')
        cmd.append('-c')
        cmd.append('-o')
        cmd.append('current')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.addrobj)
        rc, out, dummy = self.module.run_command(cmd)
        out = out.rstrip()
        if rc == 0 and self.value == out:
            return True
        else:
            return False

    def set_property(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('set-addrprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property + '=' + self.value)
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)

    def reset_property(self):
        cmd = [self.module.get_bin_path('ipadm')]
        cmd.append('reset-addrprop')
        if self.temporary:
            cmd.append('-t')
        cmd.append('-p')
        cmd.append(self.property)
        cmd.append(self.addrobj)
        return self.module.run_command(cmd)