from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.volumes import BoundVolume
class AnsibleHCloudVolume(AnsibleHCloud):
    represent = 'hcloud_volume'
    hcloud_volume: BoundVolume | None = None

    def _prepare_result(self):
        server_name = None
        if self.hcloud_volume.server is not None:
            server_name = to_native(self.hcloud_volume.server.name)
        return {'id': to_native(self.hcloud_volume.id), 'name': to_native(self.hcloud_volume.name), 'size': self.hcloud_volume.size, 'location': to_native(self.hcloud_volume.location.name), 'labels': self.hcloud_volume.labels, 'server': server_name, 'linux_device': to_native(self.hcloud_volume.linux_device), 'delete_protection': self.hcloud_volume.protection['delete']}

    def _get_volume(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_volume = self.client.volumes.get_by_id(self.module.params.get('id'))
            else:
                self.hcloud_volume = self.client.volumes.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_volume(self):
        self.module.fail_on_missing_params(required_params=['name', 'size'])
        params = {'name': self.module.params.get('name'), 'size': self.module.params.get('size'), 'automount': self.module.params.get('automount'), 'format': self.module.params.get('format'), 'labels': self.module.params.get('labels')}
        if self.module.params.get('server') is not None:
            params['server'] = self.client.servers.get_by_name(self.module.params.get('server'))
        elif self.module.params.get('location') is not None:
            params['location'] = self.client.locations.get_by_name(self.module.params.get('location'))
        else:
            self.module.fail_json(msg='server or location is required')
        if not self.module.check_mode:
            try:
                resp = self.client.volumes.create(**params)
                resp.action.wait_until_finished()
                [action.wait_until_finished() for action in resp.next_actions]
                delete_protection = self.module.params.get('delete_protection')
                if delete_protection is not None:
                    self._get_volume()
                    self.hcloud_volume.change_protection(delete=delete_protection).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_volume()

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

    def present_volume(self):
        self._get_volume()
        if self.hcloud_volume is None:
            self._create_volume()
        else:
            self._update_volume()

    def delete_volume(self):
        try:
            self._get_volume()
            if self.hcloud_volume is not None:
                if not self.module.check_mode:
                    if self.hcloud_volume.server is not None:
                        self.hcloud_volume.detach().wait_until_finished()
                    self.client.volumes.delete(self.hcloud_volume)
                self._mark_as_changed()
            self.hcloud_volume = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, size={'type': 'int'}, location={'type': 'str'}, server={'type': 'str'}, labels={'type': 'dict'}, automount={'type': 'bool', 'default': False}, format={'type': 'str', 'choices': ['xfs', 'ext4']}, delete_protection={'type': 'bool'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], mutually_exclusive=[['location', 'server']], supports_check_mode=True)