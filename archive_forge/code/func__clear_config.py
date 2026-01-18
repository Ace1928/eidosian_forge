from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def _clear_config(self, want, have):
    commands = []
    if want.get('name'):
        interface = 'interface ' + want['name']
    else:
        interface = 'interface ' + have['name']
    if have.get('port_priority') and have.get('port_priority') != want.get('port_priority'):
        cmd = 'lacp port-priority'
        remove_command_from_config_list(interface, cmd, commands)
    if have.get('max_bundle') and have.get('max_bundle') != want.get('max_bundle'):
        cmd = 'lacp max-bundle'
        remove_command_from_config_list(interface, cmd, commands)
    if have.get('fast_switchover'):
        cmd = 'lacp fast-switchover'
        remove_command_from_config_list(interface, cmd, commands)
    return commands