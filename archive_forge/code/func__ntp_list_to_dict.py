from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.rm_templates.ntp_global import (
def _ntp_list_to_dict(self, entry):
    servers_dict = {}
    for k, data in iteritems(entry):
        if k == 'servers':
            for value in data:
                if 'options' in value:
                    result = self._serveroptions_list_to_dict(value)
                    for res, resvalue in iteritems(result):
                        servers_dict.update({res: resvalue})
                else:
                    servers_dict.update({value['server']: value})
        else:
            for value in data:
                servers_dict.update({'ip_' + value: {k: value}})
    return servers_dict