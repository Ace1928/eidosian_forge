from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def _get_primary_ip(self):
    try:
        if self.module.params.get('id') is not None:
            self.hcloud_primary_ip = self.client.primary_ips.get_by_id(self.module.params.get('id'))
        else:
            self.hcloud_primary_ip = self.client.primary_ips.get_by_name(self.module.params.get('name'))
    except HCloudException as exception:
        self.fail_json_hcloud(exception)