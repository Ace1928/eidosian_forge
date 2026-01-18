from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _update_next_hop(self, want, have, opr=True):
    """
        This function gets the difference for next_hops list and
        form the commands to delete the attributes which are present in have but not in want.
        :param want:
        :param have:
        :return: list of commands
        """
    commands = []
    want_copy = deepcopy(remove_empties(want))
    have_copy = deepcopy(remove_empties(have))
    diff_next_hops = get_lst_diff_for_dicts(have_copy, want_copy, 'next_hops')
    if diff_next_hops:
        for hop in diff_next_hops:
            for element in hop:
                if element == 'forward_router_address':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', value=hop[element], remove=True))
                elif element == 'enabled':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'], value='disable', remove=True))
                elif element == 'admin_distance':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'] + ' ' + 'distance', value=str(hop[element]), remove=True))
                elif element == 'interface':
                    commands.append(self._compute_command(dest=want['dest'], key='next-hop', attrib=hop['forward_router_address'] + ' ' + 'next-hop-interface', value=hop[element], remove=True))
    return commands