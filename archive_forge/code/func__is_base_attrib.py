from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.facts.facts import Facts
from ansible_collections.vyos.vyos.plugins.module_utils.network.vyos.utils.utils import (
def _is_base_attrib(self, key):
    """
        This function checks whether key is present in predefined
        based attribute set.
        :param key:
        :return: True/False.
        """
    r_set = ('p2p', 'ipsec', 'log', 'action', 'fragment', 'protocol', 'disable', 'description', 'mac_address', 'default_action', 'enable_default_log')
    return True if key in r_set else False