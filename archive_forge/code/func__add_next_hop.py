from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_next_hop(self, want, have, opr=True):
    """
        This function gets the diff for next hop specific attributes
        and form the commands to add attributes which are present in want but not in have.
        :param want:
        :param have:
        :return: list of commands.
        """
    commands = []
    want_copy = deepcopy(remove_empties(want))
    have_copy = deepcopy(remove_empties(have))
    if not opr:
        diff_next_hops = get_lst_same_for_dicts(want_copy, have_copy, 'next_hops')
    else:
        diff_next_hops = get_lst_diff_for_dicts(want_copy, have_copy, 'next_hops')
    if diff_next_hops:
        for hop in diff_next_hops:
            for element in hop:
                if element == 'forward_router_address':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', value=hop[element], opr=opr))
                elif element == 'enabled' and (not hop[element]):
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'], value='disable', opr=opr))
                elif element == 'admin_distance':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'] + ' ' + 'distance', value=str(hop[element]), opr=opr))
                elif element == 'interface':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'] + ' ' + 'next-hop-interface', value=hop[element], opr=opr))
    return commands