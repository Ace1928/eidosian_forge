from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_area_type(self, want, have, attr, cmd, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for area_types attributes.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: command to prepend.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h_type = {}
    w_type = want.get(attr) or []
    if have:
        h_type = have.get(attr) or {}
    if not opr and (not h_type):
        commands.append(cmd + attr.replace('_', '-'))
    elif w_type:
        key = 'normal'
        if opr and key in w_type.keys() and (not _is_w_same(w_type, h_type, key)):
            if not w_type[key] and h_type and h_type[key]:
                commands.append(cmd.replace('set', 'delete') + attr.replace('_', '-') + ' ' + key)
            elif w_type[key]:
                commands.append(cmd + attr.replace('_', '-') + ' ' + key)
        elif not opr and key in w_type.keys() and (not (h_type and key in h_type.keys())):
            commands.append(cmd + want['area'] + ' ' + attr.replace('_', '-'))
        a_type = {'nssa': ('set', 'default_cost', 'no_summary', 'translate'), 'stub': ('set', 'default_cost', 'no_summary')}
        for key in a_type:
            w_area = want[attr].get(key) or {}
            h_area = {}
            if w_area:
                if h_type and key in h_type.keys():
                    h_area = h_type.get(key) or {}
                for item, val in iteritems(w_type[key]):
                    if opr and item in a_type[key] and (not _is_w_same(w_type[key], h_area, item)):
                        if item == 'set' and val:
                            commands.append(cmd + attr.replace('_', '-') + ' ' + key)
                        elif not val and h_area and h_area[item]:
                            commands.append(cmd.replace('set', 'delete') + attr.replace('_', '-') + ' ' + key)
                        elif item != 'set':
                            commands.append(cmd + attr.replace('_', '-') + ' ' + key + ' ' + item.replace('_', '-') + ' ' + str(val))
                    elif not opr and item in a_type[key] and (not (h_type and key in h_type)):
                        if item == 'set':
                            commands.append(cmd + attr.replace('_', '-') + ' ' + key)
                        else:
                            commands.append(cmd + want['area'] + ' ' + attr.replace('_', '-') + ' ' + key + ' ' + item.replace('_', '-'))
    return commands