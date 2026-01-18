from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
def _update_primary_ip(self):
    try:
        labels = self.module.params.get('labels')
        if labels is not None and labels != self.hcloud_primary_ip.labels:
            if not self.module.check_mode:
                self.hcloud_primary_ip.update(labels=labels)
            self._mark_as_changed()
        delete_protection = self.module.params.get('delete_protection')
        if delete_protection is not None and delete_protection != self.hcloud_primary_ip.protection['delete']:
            if not self.module.check_mode:
                self.hcloud_primary_ip.change_protection(delete=delete_protection).wait_until_finished()
            self._mark_as_changed()
        self._get_primary_ip()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)