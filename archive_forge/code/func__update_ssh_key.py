from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
def _update_ssh_key(self):
    name = self.module.params.get('name')
    if name is not None and self.hcloud_ssh_key.name != name:
        self.module.fail_on_missing_params(required_params=['id'])
        if not self.module.check_mode:
            self.hcloud_ssh_key.update(name=name)
        self._mark_as_changed()
    labels = self.module.params.get('labels')
    if labels is not None and self.hcloud_ssh_key.labels != labels:
        if not self.module.check_mode:
            self.hcloud_ssh_key.update(labels=labels)
        self._mark_as_changed()
    self._get_ssh_key()