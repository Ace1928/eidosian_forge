from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
def _delete_access_rules(self, want, have, opr=False):
    """
        This function forms the delete commands based on the 'opr' type
        for 'access_rules' attributes.
        :param want: desired config.
        :param have: target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h_rules = {}
    w_rs = deepcopy(remove_empties(want))
    w_rules = w_rs.get('access_rules') or []
    if have:
        h_rs = deepcopy(remove_empties(have))
        h_rules = h_rs.get('access_rules') or []
    if not w_rules and h_rules:
        commands.append(self._compute_command(name=want['name'], opr=opr))
    if w_rules:
        for w in w_rules:
            h = search_obj_in_list(w['afi'], h_rules, key='afi')
            commands.extend(self._delete_rules(want['name'], w, h))
    return commands