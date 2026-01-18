from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_areas(self, attr, want, have, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for ospf area attributes.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    h_lst = {}
    w_lst = want.get(attr) or []
    l_set = ('area_id', 'shortcut', 'authentication')
    if have:
        h_lst = have.get(attr) or []
    if not opr and (not h_lst):
        commands.append(self._form_attr_cmd(attr='area', opr=opr))
    elif w_lst:
        for w_area in w_lst:
            cmd = self._compute_command(key='area', attr=_bool_to_str(w_area['area_id']), opr=opr) + ' '
            h_area = self.search_obj_in_have(h_lst, w_area, 'area_id')
            if not opr and (not h_area):
                commands.append(self._form_attr_cmd(key='area', attr=w_area['area_id'], opr=opr))
            else:
                for key, val in iteritems(w_area):
                    if opr and key in l_set and (not _is_w_same(w_area, h_area, key)):
                        if key == 'area_id':
                            commands.append(self._form_attr_cmd(attr='area', val=_bool_to_str(val), opr=opr))
                        else:
                            commands.append(cmd + key + ' ' + _bool_to_str(val).replace('_', '-'))
                    elif not opr and key in l_set:
                        if key == 'area_id' and (not _in_target(h_area, key)):
                            commands.append(cmd)
                            continue
                        if key != 'area_id' and (not _in_target(h_area, key)):
                            commands.append(cmd + val + ' ' + key)
                    elif key == 'area_type':
                        commands.extend(self._render_area_type(w_area, h_area, key, cmd, opr))
                    elif key == 'network':
                        commands.extend(self._render_list_param(key, w_area, h_area, cmd, opr))
                    elif key == 'range':
                        commands.extend(self._render_list_dict_param(key, w_area, h_area, cmd, opr))
                    elif key == 'virtual_link':
                        commands.extend(self._render_vlink(key, w_area, h_area, cmd, opr))
    return commands