from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.ssh_keys import BoundSSHKey
def _create_ssh_key(self):
    self.module.fail_on_missing_params(required_params=['name', 'public_key'])
    params = {'name': self.module.params.get('name'), 'public_key': self.module.params.get('public_key'), 'labels': self.module.params.get('labels')}
    if not self.module.check_mode:
        try:
            self.client.ssh_keys.create(**params)
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
    self._mark_as_changed()
    self._get_ssh_key()