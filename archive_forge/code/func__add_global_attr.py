from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_global_attr(self, w, h, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for firewall_global attributes.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    w_fg = deepcopy(remove_empties(w))
    l_set = ('config_trap', 'validation', 'log_martians', 'syn_cookies', 'twa_hazards_protection')
    if w_fg:
        for key, val in iteritems(w_fg):
            if opr and key in l_set and (not (h and self._is_w_same(w_fg, h, key))):
                commands.append(self._form_attr_cmd(attr=key, val=self._bool_to_str(val), opr=opr))
            elif not opr:
                if key and self._is_del(l_set, h):
                    commands.append(self._form_attr_cmd(attr=key, key=self._bool_to_str(val), opr=opr))
                    continue
                if key in l_set and (not (h and self._in_target(h, key))) and (not self._is_del(l_set, h)):
                    commands.append(self._form_attr_cmd(attr=key, val=self._bool_to_str(val), opr=opr))
            else:
                commands.extend(self._render_attr_config(w_fg, h, key, opr))
    return commands