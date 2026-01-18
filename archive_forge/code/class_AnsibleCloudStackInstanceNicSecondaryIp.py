from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackInstanceNicSecondaryIp(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackInstanceNicSecondaryIp, self).__init__(module)
        self.vm_guest_ip = self.module.params.get('vm_guest_ip')
        self.nic = None
        self.returns = {'ipaddress': 'ip_address', 'macaddress': 'mac_address', 'netmask': 'netmask'}

    def get_nic(self):
        if self.nic:
            return self.nic
        args = {'virtualmachineid': self.get_vm(key='id'), 'networkid': self.get_network(key='id')}
        nics = self.query_api('listNics', **args)
        if nics:
            self.nic = nics['nic'][0]
            return self.nic
        self.fail_json(msg='NIC for VM %s in network %s not found' % (self.get_vm(key='name'), self.get_network(key='name')))

    def get_secondary_ip(self):
        nic = self.get_nic()
        if self.vm_guest_ip:
            secondary_ips = nic.get('secondaryip') or []
            for secondary_ip in secondary_ips:
                if secondary_ip['ipaddress'] == self.vm_guest_ip:
                    return secondary_ip
        return None

    def present_nic_ip(self):
        nic = self.get_nic()
        if not self.get_secondary_ip():
            self.result['changed'] = True
            args = {'nicid': nic['id'], 'ipaddress': self.vm_guest_ip}
            if not self.module.check_mode:
                res = self.query_api('addIpToNic', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    nic = self.poll_job(res, 'nicsecondaryip')
                    self.vm_guest_ip = nic['ipaddress']
        return nic

    def absent_nic_ip(self):
        nic = self.get_nic()
        secondary_ip = self.get_secondary_ip()
        if secondary_ip:
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('removeIpFromNic', id=secondary_ip['id'])
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'nicsecondaryip')
        return nic

    def get_result(self, resource):
        super(AnsibleCloudStackInstanceNicSecondaryIp, self).get_result(resource)
        if resource and (not self.module.params.get('network')):
            self.module.params['network'] = resource.get('networkid')
        self.result['network'] = self.get_network(key='name')
        self.result['vm'] = self.get_vm(key='name')
        self.result['vm_guest_ip'] = self.vm_guest_ip
        return self.result