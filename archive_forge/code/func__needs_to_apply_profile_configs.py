from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _needs_to_apply_profile_configs(self):
    return self._needs_to_change_profile_config('config') or self._needs_to_change_profile_config('description') or self._needs_to_change_profile_config('devices')