from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_r_base_attrib(self, afi, name, attr, rule, opr=True):
    """
        This function forms the command for 'rules' attributes which doesn't
        have further sub attributes.
        :param afi: address type.
        :param name: rule-set name
        :param attrib: attribute name
        :param rule: rule config dictionary.
        :param opr: True/False.
        :return: generated command.
        """
    if attr == 'number':
        command = self._compute_command(afi=afi, name=name, number=rule['number'], opr=opr)
    else:
        command = self._compute_command(afi=afi, name=name, number=rule['number'], attrib=attr, value=rule[attr], opr=opr)
    return command