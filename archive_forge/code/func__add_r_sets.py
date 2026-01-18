from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_r_sets(self, afi, want, have, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for rule-sets attributes.
        :param afi: address type.
        :param want: desired config.
        :param have: target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    l_set = ('description', 'default_action', 'enable_default_log')
    h_rs = {}
    h_rules = {}
    w_rs = deepcopy(remove_empties(want))
    w_rules = w_rs.pop('rules', None)
    if have:
        h_rs = deepcopy(remove_empties(have))
        h_rules = h_rs.pop('rules', None)
    if w_rs:
        for key, val in iteritems(w_rs):
            if opr and key in l_set and (not (h_rs and self._is_w_same(w_rs, h_rs, key))):
                if key == 'enable_default_log':
                    if val and (not h_rs or key not in h_rs or (not h_rs[key])):
                        commands.append(self._add_rs_base_attrib(afi, want['name'], key, w_rs))
                else:
                    commands.append(self._add_rs_base_attrib(afi, want['name'], key, w_rs))
            elif not opr and key in l_set:
                if key == 'enable_default_log' and val and h_rs and (key not in h_rs or not h_rs[key]):
                    commands.append(self._add_rs_base_attrib(afi, want['name'], key, w_rs, opr))
                elif not (h_rs and self._in_target(h_rs, key)):
                    commands.append(self._add_rs_base_attrib(afi, want['name'], key, w_rs, opr))
        commands.extend(self._add_rules(afi, want['name'], w_rules, h_rules, opr))
    if h_rules:
        have['rules'] = h_rules
    if w_rules:
        want['rules'] = w_rules
    return commands