from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_nested_dict_param(self, attr, want, have, opr=True):
    """
        This function forms the set/delete commands based on the 'opr' type
        for attributes with in desired nested dicts.
        :param attr: attribute name.
        :param w: the desired config.
        :param h: the target config.
        :param cmd: commands to be prepend.
        :param opr: True/False.
        :return: generated commands list.
        """
    commands = []
    attr_dict = {'default_information': 'originate', 'max_metric': 'router_lsa'}
    leaf_dict = {'default_information': ('always', 'metric', 'metric_type', 'route_map'), 'max_metric': ('administrative', 'on_startup', 'on_shutdown')}
    h = {}
    w = want.get(attr) or {}
    if have:
        h = have.get(attr) or {}
    if not opr and (not h):
        commands.append(self._form_attr_cmd(attr=attr, opr=opr))
    elif w:
        key = attr_dict[attr]
        w_attrib = want[attr].get(key) or {}
        cmd = self._compute_command(opr=opr)
        h_attrib = {}
        if w_attrib:
            leaf = leaf_dict[attr]
            if h and key in h.keys():
                h_attrib = h.get(key) or {}
            for item, val in iteritems(w[key]):
                if opr and item in leaf and (not _is_w_same(w[key], h_attrib, item)):
                    if item in ('administrative', 'always') and val:
                        commands.append(cmd + attr.replace('_', '-') + ' ' + key.replace('_', '-') + ' ' + item.replace('_', '-'))
                    elif item not in ('administrative', 'always'):
                        commands.append(cmd + attr.replace('_', '-') + ' ' + key.replace('_', '-') + ' ' + item.replace('_', '-') + ' ' + str(val))
                elif not opr and item in leaf and (not _in_target(h_attrib, item)):
                    commands.append(cmd + attr + ' ' + item)
    return commands