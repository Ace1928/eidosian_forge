from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _is_ip_route_exist(self, routes, type='route'):
    """
        This functions checks for the type of route.
        :param routes:
        :param type:
        :return: True/False
        """
    for r in routes:
        if type == self.get_route_type(r['dest']):
            return True
    return False