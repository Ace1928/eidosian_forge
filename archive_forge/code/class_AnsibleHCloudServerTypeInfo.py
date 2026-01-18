from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.server_types import BoundServerType
class AnsibleHCloudServerTypeInfo(AnsibleHCloud):
    represent = 'hcloud_server_type_info'
    hcloud_server_type_info: list[BoundServerType] | None = None

    def _prepare_result(self):
        tmp = []
        for server_type in self.hcloud_server_type_info:
            if server_type is not None:
                tmp.append({'id': to_native(server_type.id), 'name': to_native(server_type.name), 'description': to_native(server_type.description), 'cores': server_type.cores, 'memory': server_type.memory, 'disk': server_type.disk, 'storage_type': to_native(server_type.storage_type), 'cpu_type': to_native(server_type.cpu_type), 'architecture': to_native(server_type.architecture), 'included_traffic': server_type.included_traffic, 'deprecation': {'announced': server_type.deprecation.announced.isoformat(), 'unavailable_after': server_type.deprecation.unavailable_after.isoformat()} if server_type.deprecation is not None else None})
        return tmp

    def get_server_types(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_server_type_info = [self.client.server_types.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_server_type_info = [self.client.server_types.get_by_name(self.module.params.get('name'))]
            else:
                self.hcloud_server_type_info = self.client.server_types.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)