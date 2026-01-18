from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class FcNetworkInfoModule(OneViewModuleBase):

    def __init__(self):
        argument_spec = dict(name=dict(required=False, type='str'), params=dict(required=False, type='dict'))
        super(FcNetworkInfoModule, self).__init__(additional_arg_spec=argument_spec, supports_check_mode=True)

    def execute_module(self):
        if self.module.params['name']:
            fc_networks = self.oneview_client.fc_networks.get_by('name', self.module.params['name'])
        else:
            fc_networks = self.oneview_client.fc_networks.get_all(**self.facts_params)
        return dict(changed=False, fc_networks=fc_networks)