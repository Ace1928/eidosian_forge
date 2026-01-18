from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import APIException, HCloudException
from ..module_utils.vendor.hcloud.networks import BoundNetwork
from ..module_utils.vendor.hcloud.servers import BoundServer, PrivateNet
class AnsibleHCloudServerNetwork(AnsibleHCloud):
    represent = 'hcloud_server_network'
    hcloud_network: BoundNetwork | None = None
    hcloud_server: BoundServer | None = None
    hcloud_server_network: PrivateNet | None = None

    def _prepare_result(self):
        return {'network': to_native(self.hcloud_network.name), 'server': to_native(self.hcloud_server.name), 'ip': to_native(self.hcloud_server_network.ip), 'alias_ips': self.hcloud_server_network.alias_ips}

    def _get_server_and_network(self):
        try:
            self.hcloud_network = self._client_get_by_name_or_id('networks', self.module.params.get('network'))
            self.hcloud_server = self._client_get_by_name_or_id('servers', self.module.params.get('server'))
            self.hcloud_server_network = None
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _get_server_network(self):
        for private_net in self.hcloud_server.private_net:
            if private_net.network.id == self.hcloud_network.id:
                self.hcloud_server_network = private_net

    def _create_server_network(self):
        params = {'network': self.hcloud_network}
        if self.module.params.get('ip') is not None:
            params['ip'] = self.module.params.get('ip')
        if self.module.params.get('alias_ips') is not None:
            params['alias_ips'] = self.module.params.get('alias_ips')
        if not self.module.check_mode:
            try:
                self.hcloud_server.attach_to_network(**params).wait_until_finished()
            except HCloudException as exception:
                self.fail_json_hcloud(exception)
        self._mark_as_changed()
        self._get_server_and_network()
        self._get_server_network()

    def _update_server_network(self):
        params = {'network': self.hcloud_network}
        alias_ips = self.module.params.get('alias_ips')
        if alias_ips is not None and sorted(self.hcloud_server_network.alias_ips) != sorted(alias_ips):
            params['alias_ips'] = alias_ips
            if not self.module.check_mode:
                try:
                    self.hcloud_server.change_alias_ips(**params).wait_until_finished()
                except APIException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self._get_server_and_network()
        self._get_server_network()

    def present_server_network(self):
        self._get_server_and_network()
        self._get_server_network()
        if self.hcloud_server_network is None:
            self._create_server_network()
        else:
            self._update_server_network()

    def delete_server_network(self):
        self._get_server_and_network()
        self._get_server_network()
        if self.hcloud_server_network is not None and self.hcloud_server is not None:
            if not self.module.check_mode:
                try:
                    self.hcloud_server.detach_from_network(self.hcloud_server_network.network).wait_until_finished()
                except HCloudException as exception:
                    self.fail_json_hcloud(exception)
            self._mark_as_changed()
        self.hcloud_server_network = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(network={'type': 'str', 'required': True}, server={'type': 'str', 'required': True}, ip={'type': 'str'}, alias_ips={'type': 'list', 'elements': 'str'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), supports_check_mode=True)