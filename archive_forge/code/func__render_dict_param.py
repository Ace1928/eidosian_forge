from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_dict_param(self, attr, want, have, opr=True):
    """
        This function generate the commands for dictionary elements.
        :param attr: attribute name.
        :param w: the desired configuration.
        :param h: the target config.
        :param opr: True/False.
        :return: generated list of commands.
        """
    commands = []
    h = {}
    if have:
        h = have.get(attr) or {}
    if not opr and (not h):
        commands.append(self._form_attr_cmd(attr=attr, opr=opr))
    elif want[attr]:
        leaf_dict = {'auto_cost': 'reference_bandwidth', 'mpls_te': ('enabled', 'router_address'), 'parameters': ('router_id', 'abr_type', 'opaque_lsa', 'rfc1583_compatibility')}
        leaf = leaf_dict[attr]
        for item, value in iteritems(want[attr]):
            if opr and item in leaf and (not _is_w_same(want[attr], h, item)):
                if item == 'enabled':
                    item = 'enable'
                if item in ('opaque_lsa', 'enable', 'rfc1583_compatibility'):
                    commands.append(self._form_attr_cmd(key=attr, attr=item, opr=opr))
                else:
                    commands.append(self._form_attr_cmd(key=attr, attr=item, val=value, opr=opr))
            elif not opr and item in leaf and (not _in_target(h, item)):
                if item == 'enabled':
                    commands.append(self._form_attr_cmd(key=attr, attr='enable', opr=opr))
                else:
                    commands.append(self._form_attr_cmd(key=attr, attr=item, opr=opr))
    return commands