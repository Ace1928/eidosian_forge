from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackNetworkAcl(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackNetworkAcl, self).__init__(module)

    def get_network_acl(self):
        args = {'name': self.module.params.get('name'), 'vpcid': self.get_vpc(key='id')}
        network_acls = self.query_api('listNetworkACLLists', **args)
        if network_acls:
            return network_acls['networkacllist'][0]
        return None

    def present_network_acl(self):
        network_acl = self.get_network_acl()
        if not network_acl:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'description': self.get_or_fallback('description', 'name'), 'vpcid': self.get_vpc(key='id')}
            if not self.module.check_mode:
                res = self.query_api('createNetworkACLList', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    network_acl = self.poll_job(res, 'networkacllist')
        return network_acl

    def absent_network_acl(self):
        network_acl = self.get_network_acl()
        if network_acl:
            self.result['changed'] = True
            args = {'id': network_acl['id']}
            if not self.module.check_mode:
                res = self.query_api('deleteNetworkACLList', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(res, 'networkacllist')
        return network_acl