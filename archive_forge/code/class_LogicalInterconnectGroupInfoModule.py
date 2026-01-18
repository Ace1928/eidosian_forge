from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class LogicalInterconnectGroupInfoModule(OneViewModuleBase):

    def __init__(self):
        argument_spec = dict(name=dict(type='str'), params=dict(type='dict'))
        super(LogicalInterconnectGroupInfoModule, self).__init__(additional_arg_spec=argument_spec, supports_check_mode=True)

    def execute_module(self):
        if self.module.params.get('name'):
            ligs = self.oneview_client.logical_interconnect_groups.get_by('name', self.module.params['name'])
        else:
            ligs = self.oneview_client.logical_interconnect_groups.get_all(**self.facts_params)
        return dict(changed=False, logical_interconnect_groups=ligs)