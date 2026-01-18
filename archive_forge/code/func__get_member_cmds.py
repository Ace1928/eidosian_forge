from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.facts.facts import Facts
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import dict_to_set
def _get_member_cmds(self, member_dict, prefix=''):
    cmd = ''
    if prefix:
        prefix = prefix + ' '
    member_vni = member_dict.get('vni')
    member_evi = member_dict.get('evi')
    if member_evi:
        cmd = prefix + 'member evpn-instance {0} vni {1}'.format(member_evi, member_vni)
    elif member_vni:
        cmd = prefix + 'member vni {0}'.format(member_vni)
    return cmd