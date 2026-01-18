from __future__ import absolute_import, division, print_function
import copy
import datetime
import os
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.lxd import LXDClient, LXDClientException
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _apply_instance_configs(self):
    old_metadata = copy.deepcopy(self.old_instance_json).get('metadata', None) or {}
    body_json = {}
    for param in set(CONFIG_PARAMS) - set(CONFIG_CREATION_PARAMS):
        if param in old_metadata:
            body_json[param] = old_metadata[param]
        if self._needs_to_change_instance_config(param):
            if param == 'config':
                body_json['config'] = body_json.get('config', None) or {}
                for k, v in self.config['config'].items():
                    body_json['config'][k] = v
            else:
                body_json[param] = self.config[param]
    self.diff['after']['instance'] = body_json
    url = '{0}/{1}'.format(self.api_endpoint, self.name)
    if self.project:
        url = '{0}?{1}'.format(url, urlencode(dict(project=self.project)))
    if not self.module.check_mode:
        self.client.do('PUT', url, body_json=body_json)
    self.actions.append('apply_instance_configs')