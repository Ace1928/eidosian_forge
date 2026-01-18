from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.frr.frr.plugins.module_utils.network.frr.providers.providers import (
def _render_description(self, item, config=None):
    cmd = 'neighbor %s description %s' % (item['neighbor'], item['description'])
    if not config or cmd not in config:
        return cmd