from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.address_family import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.cli.config.bgp.neighbors import (
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_neighbors(self, config):
    """generate bgp neighbor configuration"""
    return Neighbors(self.params).render(config)