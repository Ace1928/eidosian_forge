from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_vpn_gateway(self):
    vpn_gateway = self.get_vpn_gateway()
    if vpn_gateway:
        self.result['changed'] = True
        args = {'id': vpn_gateway['id']}
        if not self.module.check_mode:
            res = self.query_api('deleteVpnGateway', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'vpngateway')
    return vpn_gateway