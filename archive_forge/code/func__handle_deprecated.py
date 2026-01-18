from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ospfv2 import (
def _handle_deprecated(self, config):
    if config.get('passive_interface'):
        passive_interfaces = config.get('passive_interfaces', {})
        interface = passive_interfaces.get('interface', {})
        name_list = interface.get('name', [])
        if not name_list:
            name_list.append(config['passive_interface'])
        else:
            name_list.extend(config['passive_interface'])
        del config['passive_interface']
    return config