from __future__ import absolute_import, division, print_function
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.frr.frr.plugins.module_utils.network.frr.frr import (
def _parse_daemons(self, data):
    match = re.search('Memory statistics for (\\w+)', data, re.M)
    if match:
        return match.group(1)