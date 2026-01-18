from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.firewalls import (
from ..module_utils.vendor.hcloud.servers import BoundServer
def absent_firewall_resources(self):
    self._get_firewall()
    resources = self._diff_firewall_resources(lambda to_remove, before: to_remove in before)
    if resources:
        if not self.module.check_mode:
            actions = self.hcloud_firewall_resource.remove_from_resources(resources=resources)
            for action in actions:
                action.wait_until_finished()
            self.hcloud_firewall_resource.reload()
        self._mark_as_changed()