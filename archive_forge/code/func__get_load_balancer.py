from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.load_balancers import (
def _get_load_balancer(self):
    try:
        self.hcloud_load_balancer = self._client_get_by_name_or_id('load_balancers', self.module.params.get('load_balancer'))
        self._get_load_balancer_service()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)