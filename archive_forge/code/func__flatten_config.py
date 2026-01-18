from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.argspec.bgp_neighbor_address_family.bgp_neighbor_address_family import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.rm_templates.bgp_neighbor_address_family import (
def _flatten_config(self, data):
    """Flatten contexts in the BGP
            running-config for easier parsing.
            Only neighbor AF contexts are returned.
        :param data: str
        :returns: flattened running config
        """
    data = data.split('\n')
    nbr_af_cxt = []
    context = ''
    cur_vrf = ''
    cur_nbr_indent = None
    in_nbr_cxt = False
    in_af = False
    nbr_af_cxt.append(data[0])
    for x in data:
        cur_indent = len(x) - len(x.lstrip())
        x = x.strip()
        if x.startswith('vrf'):
            cur_vrf = x + ' '
            in_nbr_cxt = False
        elif x.startswith('neighbor'):
            in_nbr_cxt = True
            in_af = False
            cur_nbr_indent = cur_indent
            context = x
            if cur_vrf:
                context = cur_vrf + context
        elif in_nbr_cxt and cur_indent > cur_nbr_indent:
            if x.startswith('address-family'):
                in_af = True
                x = context + ' ' + x
            if in_af:
                nbr_af_cxt.append(x)
        else:
            in_nbr_cxt = False
    return nbr_af_cxt