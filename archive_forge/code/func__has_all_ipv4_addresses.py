from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
@staticmethod
def _has_all_ipv4_addresses(addresses):
    return len(addresses) > 0 and all((len(v) > 0 for v in addresses.values()))