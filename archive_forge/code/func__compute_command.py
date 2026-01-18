from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _compute_command(self, key=None, attr=None, val=None, remove=False, opr=True):
    """
        This function construct the add/delete command based on passed attributes.
        :param key: parent key.
        :param attr: attribute name
        :param value: value
        :param opr: True/False.
        :return: generated command.
        """
    if remove or not opr:
        cmd = 'delete protocols ospf '
    else:
        cmd = 'set protocols ospf '
    if key:
        cmd += key.replace('_', '-') + ' '
    if attr:
        cmd += attr.replace('_', '-')
    if val:
        cmd += " '" + str(val) + "'"
    return cmd