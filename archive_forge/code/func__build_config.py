from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _build_config(self):
    self.config = {}
    for attr in CONFIG_PARAMS:
        param_val = self.module.params.get(attr, None)
        if param_val is not None:
            self.config[attr] = param_val