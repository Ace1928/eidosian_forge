from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def _create_firewall(self):
    self.module.fail_on_missing_params(required_params=['name'])
    params = {'name': self.module.params.get('name'), 'labels': self.module.params.get('labels')}
    rules = self.module.params.get('rules')
    if rules is not None:
        params['rules'] = [FirewallRule(direction=rule['direction'], protocol=rule['protocol'], source_ips=rule['source_ips'] if rule['source_ips'] is not None else [], destination_ips=rule['destination_ips'] if rule['destination_ips'] is not None else [], port=rule['port'], description=rule['description']) for rule in rules]
    if not self.module.check_mode:
        try:
            self.client.firewalls.create(**params)
        except HCloudException as exception:
            self.fail_json_hcloud(exception, params=params)
    self._mark_as_changed()
    self._get_firewall()