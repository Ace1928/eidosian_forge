from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _render_attr_config(self, w, h, key, opr=False):
    """
        This function invoke the function to extend commands
        based on the key.
        :param w: the desired configuration.
        :param h: the current configuration.
        :param key: attribute name
        :param opr: operation
        :return: list of commands
        """
    commands = []
    if key == 'ping':
        commands.extend(self._render_ping(key, w, h, opr=opr))
    elif key == 'group':
        commands.extend(self._render_group(key, w, h, opr=opr))
    elif key == 'state_policy':
        commands.extend(self._render_state_policy(key, w, h, opr=opr))
    elif key == 'route_redirects':
        commands.extend(self._render_route_redirects(key, w, h, opr=opr))
    return commands