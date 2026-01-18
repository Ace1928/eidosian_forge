from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _add_p2p(self, attr, w, h, cmd, opr):
    """
        This function forms the set/delete commands based on the 'opr' type
        for p2p applications attributes.
        :param want: desired config.
        :param have: target config.
        :return: generated commands list.
        """
    commands = []
    have = []
    if w:
        want = w.get(attr) or []
    if h:
        have = h.get(attr) or []
    if want:
        if opr:
            applications = list_diff_want_only(want, have)
            for app in applications:
                commands.append(cmd + (' ' + attr + ' ' + app['application']))
        elif not opr and have:
            applications = list_diff_want_only(want, have)
            for app in applications:
                commands.append(cmd + (' ' + attr + ' ' + app['application']))
    return commands