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
def _set_rescue_mode(self, rescue_mode):
    if self.module.params.get('ssh_keys'):
        resp = self.hcloud_server.enable_rescue(type=rescue_mode, ssh_keys=[self.client.ssh_keys.get_by_name(ssh_key_name).id for ssh_key_name in self.module.params.get('ssh_keys')])
    else:
        resp = self.hcloud_server.enable_rescue(type=rescue_mode)
    resp.action.wait_until_finished()
    self.result['root_password'] = resp.root_password