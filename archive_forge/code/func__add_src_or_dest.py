from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_src_or_dest(self, attr, w, h, cmd, opr=True):
    """
        This function forms the commands for 'src/dest' attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: base config.
        :param h: target config.
        :param cmd: commands to be prepend.
        :return: generated list of commands.
        """
    commands = []
    h_group = {}
    g_set = ('port_group', 'address_group', 'network_group')
    if w[attr]:
        keys = ('address', 'mac_address', 'port')
        for key in keys:
            if opr and key in w[attr].keys() and (not (h and attr in h.keys() and self._is_w_same(w[attr], h[attr], key))):
                commands.append(cmd + (' ' + attr + ' ' + key.replace('_', '-') + ' ' + w[attr].get(key)))
            elif not opr and key in w[attr].keys() and (not (h and attr in h.keys() and self._in_target(h[attr], key))):
                commands.append(cmd + (' ' + attr + ' ' + key))
        key = 'group'
        group = w[attr].get(key) or {}
        if group:
            h_group = {}
            if h and h.get(attr) and (key in h[attr].keys()):
                h_group = h[attr].get(key)
            for item, val in iteritems(group):
                if val:
                    if opr and item in g_set and (not (h_group and self._is_w_same(group, h_group, item))):
                        commands.append(cmd + (' ' + attr + ' ' + key + ' ' + item.replace('_', '-') + ' ' + val))
                    elif not opr and item in g_set and (not (h_group and self._in_target(h_group, item))):
                        commands.append(cmd + (' ' + attr + ' ' + key + ' ' + item.replace('_', '-')))
    return commands