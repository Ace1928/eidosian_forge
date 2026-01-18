from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_grp_mem(self, attr, w, h, opr):
    """
        This function forms the commands for group list/members attributes based on the 'opr'.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    h_grp = []
    w_grp = []
    l_set = ('name', 'description')
    if w:
        w_grp = w.get(attr) or []
    if h:
        h_grp = h.get(attr) or []
    if w_grp:
        for want in w_grp:
            h = self.search_attrib_in_have(h_grp, want, 'name')
            if 'afi' in want and want['afi'] == 'ipv6':
                cmd = self._compute_command(key='group', attr='ipv6-' + attr, opr=opr)
            else:
                cmd = self._compute_command(key='group', attr=attr, opr=opr)
            for key, val in iteritems(want):
                if val:
                    if opr and key in l_set and (not (h and self._is_w_same(want, h, key))):
                        if key == 'name':
                            commands.append(cmd + ' ' + str(val))
                        else:
                            commands.append(cmd + ' ' + want['name'] + ' ' + key + " '" + str(want[key]) + "'")
                    elif not opr and key in l_set:
                        if key == 'name' and self._is_grp_del(h, want, key):
                            commands.append(cmd + ' ' + want['name'])
                            continue
                        if not (h and self._in_target(h, key)) and (not self._is_grp_del(h, want, key)):
                            commands.append(cmd + ' ' + want['name'] + ' ' + key)
                    elif key == 'members':
                        commands.extend(self._render_ports_addrs(key, want, h, opr, cmd, want['name'], attr))
    return commands