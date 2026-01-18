from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.snmp_server import (
def _handle_deprecates(self, want):
    """Remove deprecated attributes and set the replacment"""
    if 'traps' in want:
        if 'mpls_vpn' in want['traps']:
            want['traps'] = dict_merge(want['traps'], {'mpls': {'vpn': {'enable': want['traps']['mpls_vpn']}}})
            want['traps'].pop('mpls_vpn')
        if 'envmon' in want['traps'] and 'fan' in want['traps']['envmon']:
            want['traps']['envmon']['fan_enable'] = want['traps']['envmon']['fan'].get('enable', False)
            want['traps']['envmon'].pop('fan')
    return want