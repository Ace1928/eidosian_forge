from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_vlink(self, attr, want, have, cmd=None, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for attributes with in desired list of dictionary.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: commands to be prepend.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h = []
    name = {'virtual_link': 'address'}
    leaf_dict = {'virtual_link': ('address', 'dead_interval', 'transmit_delay', 'hello_interval', 'retransmit_interval')}
    leaf = leaf_dict[attr]
    w = want.get(attr) or []
    if have:
        h = have.get(attr) or []
    if not opr and (not h):
        commands.append(cmd + attr.replace('_', '-'))
    elif w:
        for w_item in w:
            for key, val in iteritems(w_item):
                if not cmd:
                    cmd = self._compute_command(opr=opr)
                h_item = self.search_obj_in_have(h, w_item, name[attr])
                if opr and key in leaf and (not _is_w_same(w_item, h_item, key)):
                    if key in 'address':
                        commands.append(cmd + attr.replace('_', '-') + ' ' + str(val))
                    else:
                        commands.append(cmd + attr.replace('_', '-') + ' ' + w_item[name[attr]] + ' ' + key.replace('_', '-') + ' ' + str(val))
                elif not opr and key in leaf and (not _in_target(h_item, key)):
                    if key in 'address':
                        commands.append(cmd + attr.replace('_', '-') + ' ' + str(val))
                    else:
                        commands.append(cmd + attr.replace('_', '-') + ' ' + w_item[name[attr]] + ' ' + key)
                elif key == 'authentication':
                    commands.extend(self._render_vlink_auth(attr, key, w_item, h_item, w_item['address'], cmd, opr))
    return commands