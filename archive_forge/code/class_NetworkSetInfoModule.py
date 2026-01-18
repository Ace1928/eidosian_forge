from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.oneview import OneViewModuleBase
class NetworkSetInfoModule(OneViewModuleBase):
    argument_spec = dict(name=dict(type='str'), options=dict(type='list', elements='str'), params=dict(type='dict'))

    def __init__(self):
        super(NetworkSetInfoModule, self).__init__(additional_arg_spec=self.argument_spec, supports_check_mode=True)

    def execute_module(self):
        name = self.module.params.get('name')
        if 'withoutEthernet' in self.options:
            filter_by_name = '"\'name\'=\'%s\'"' % name if name else ''
            network_sets = self.oneview_client.network_sets.get_all_without_ethernet(filter=filter_by_name)
        elif name:
            network_sets = self.oneview_client.network_sets.get_by('name', name)
        else:
            network_sets = self.oneview_client.network_sets.get_all(**self.facts_params)
        return dict(changed=False, network_sets=network_sets)