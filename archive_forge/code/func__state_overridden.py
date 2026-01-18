from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _state_overridden(self, want, have):
    """The command generator when state is overridden

        :rtype: A list
        :returns: the commands necessary to migrate the current configuration
                  to the desired configuration
        """
    commands = []
    routes = self._get_routes(have)
    for r in routes:
        route_in_want = self.search_route_in_have(want, r['dest'])
        if not route_in_want:
            commands.append(self._compute_command(r['dest'], remove=True))
    routes = self._get_routes(want)
    for r in routes:
        route_in_have = self.search_route_in_have(have, r['dest'])
        commands.extend(self._state_replaced(r, route_in_have))
    return commands