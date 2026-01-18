from __future__ import annotations
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
def _get_firewall(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_firewall = self.client.firewalls.get_by_id(self.module.params.get('id'))
        elif self.module.params.get('name') is not None:
            self.hcloud_firewall = self.client.firewalls.get_by_name(self.module.params.get('name'))
    except HCloudException as exception:
        self.fail_json_hcloud(exception)