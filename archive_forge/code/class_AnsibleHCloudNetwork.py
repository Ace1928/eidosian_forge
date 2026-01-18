from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
class AnsibleHCloudNetwork(AnsibleHCloud):
    represent = 'hcloud_network'
    hcloud_network: BoundNetwork | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_network.id), 'name': to_native(self.hcloud_network.name), 'ip_range': to_native(self.hcloud_network.ip_range), 'expose_routes_to_vswitch': self.hcloud_network.expose_routes_to_vswitch, 'delete_protection': self.hcloud_network.protection['delete'], 'labels': self.hcloud_network.labels}

    def _get_network(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_network = self.client.networks.get_by_id(self.module.params.get('id'))
            else:
                self.hcloud_network = self.client.networks.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_network(self):
        self.module.fail_on_missing_params(required_params=['name', 'ip_range'])
        params = {'name': self.module.params.get('name'), 'ip_range': self.module.params.get('ip_range'), 'labels': self.module.params.get('labels')}
        expose_routes_to_vswitch = self.module.params.get('expose_routes_to_vswitch')
        if expose_routes_to_vswitch is not None:
            params['expose_routes_to_vswitch'] = expose_routes_to_vswitch
        try:
            if not self.module.check_mode:
                self.client.networks.create(**params)
                delete_protection = self.module.params.get('delete_protection')
                if delete_protection is not None:
                    self._get_network()
                    self.hcloud_network.change_protection(delete=delete_protection).wait_until_finished()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_network()

    def _update_network(self):
        try:
            labels = self.module.params.get('labels')
            if labels is not None and labels != self.hcloud_network.labels:
                if not self.module.check_mode:
                    self.hcloud_network.update(labels=labels)
                self._mark_as_changed()
            ip_range = self.module.params.get('ip_range')
            if ip_range is not None and ip_range != self.hcloud_network.ip_range:
                if not self.module.check_mode:
                    self.hcloud_network.change_ip_range(ip_range=ip_range).wait_until_finished()
                self._mark_as_changed()
            expose_routes_to_vswitch = self.module.params.get('expose_routes_to_vswitch')
            if expose_routes_to_vswitch is not None and expose_routes_to_vswitch != self.hcloud_network.expose_routes_to_vswitch:
                if not self.module.check_mode:
                    self.hcloud_network.update(expose_routes_to_vswitch=expose_routes_to_vswitch)
                self._mark_as_changed()
            delete_protection = self.module.params.get('delete_protection')
            if delete_protection is not None and delete_protection != self.hcloud_network.protection['delete']:
                if not self.module.check_mode:
                    self.hcloud_network.change_protection(delete=delete_protection).wait_until_finished()
                self._mark_as_changed()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self._get_network()

    def present_network(self):
        self._get_network()
        if self.hcloud_network is None:
            self._create_network()
        else:
            self._update_network()

    def delete_network(self):
        try:
            self._get_network()
            if self.hcloud_network is not None:
                if not self.module.check_mode:
                    self.client.networks.delete(self.hcloud_network)
                self._mark_as_changed()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)
        self.hcloud_network = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, ip_range={'type': 'str'}, expose_routes_to_vswitch={'type': 'bool'}, labels={'type': 'dict'}, delete_protection={'type': 'bool'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], supports_check_mode=True)