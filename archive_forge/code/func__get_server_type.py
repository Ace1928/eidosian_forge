from __future__ import annotations
from datetime import datetime, timedelta, timezone
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import FirewallResource
from ..module_utils.vendor.hcloud.servers import (
from ..module_utils.vendor.hcloud.ssh_keys import SSHKey
from ..module_utils.vendor.hcloud.volumes import Volume
def _get_server_type(self):
    server_type = self.client.server_types.get_by_name(self.module.params.get('server_type'))
    if server_type is None:
        try:
            server_type = self.client.server_types.get_by_id(self.module.params.get('server_type'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception, msg=f'server_type {self.module.params.get('server_type')} was not found')
    self._check_and_warn_deprecated_server(server_type)
    return server_type