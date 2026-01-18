from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.volumes import BoundVolume
def _update_volume(self):
    try:
        size = self.module.params.get('size')
        if size:
            if self.hcloud_volume.size < size:
                if not self.module.check_mode:
                    self.hcloud_volume.resize(size).wait_until_finished()
                self._mark_as_changed()
            elif self.hcloud_volume.size > size:
                self.module.warn('Shrinking of volumes is not supported')
        server_name = self.module.params.get('server')
        if server_name:
            server = self.client.servers.get_by_name(server_name)
            if self.hcloud_volume.server is None or self.hcloud_volume.server.name != server.name:
                if not self.module.check_mode:
                    automount = self.module.params.get('automount', False)
                    self.hcloud_volume.attach(server, automount=automount).wait_until_finished()
                self._mark_as_changed()
        elif self.hcloud_volume.server is not None:
            if not self.module.check_mode:
                self.hcloud_volume.detach().wait_until_finished()
            self._mark_as_changed()
        labels = self.module.params.get('labels')
        if labels is not None and labels != self.hcloud_volume.labels:
            if not self.module.check_mode:
                self.hcloud_volume.update(labels=labels)
            self._mark_as_changed()
        delete_protection = self.module.params.get('delete_protection')
        if delete_protection is not None and delete_protection != self.hcloud_volume.protection['delete']:
            if not self.module.check_mode:
                self.hcloud_volume.change_protection(delete=delete_protection).wait_until_finished()
            self._mark_as_changed()
        self._get_volume()
    except HCloudException as exception:
        self.fail_json_hcloud(exception)