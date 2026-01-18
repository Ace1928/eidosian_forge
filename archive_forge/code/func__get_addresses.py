from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _get_addresses(self):
    try:
        due = datetime.datetime.now() + datetime.timedelta(seconds=self.timeout)
        while datetime.datetime.now() < due:
            time.sleep(1)
            addresses = self._instance_ipv4_addresses()
            if self._has_all_ipv4_addresses(addresses) or self.module.check_mode:
                self.addresses = addresses
                return
    except LXDClientException as e:
        e.msg = 'timeout for getting IPv4 addresses'
        raise