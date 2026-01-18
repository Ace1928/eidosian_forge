from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.datacenters import BoundDatacenter
class AnsibleHCloudDatacenterInfo(AnsibleHCloud):
    represent = 'hcloud_datacenter_info'
    hcloud_datacenter_info: list[BoundDatacenter] | None = None

    def _prepare_result(self):
        tmp = []
        for datacenter in self.hcloud_datacenter_info:
            if datacenter is None:
                continue
            tmp.append({'id': to_native(datacenter.id), 'name': to_native(datacenter.name), 'description': to_native(datacenter.description), 'location': to_native(datacenter.location.name), 'server_types': {'available': [o.id for o in datacenter.server_types.available], 'available_for_migration': [o.id for o in datacenter.server_types.available_for_migration], 'supported': [o.id for o in datacenter.server_types.supported]}})
        return tmp

    def get_datacenters(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_datacenter_info = [self.client.datacenters.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_datacenter_info = [self.client.datacenters.get_by_name(self.module.params.get('name'))]
            else:
                self.hcloud_datacenter_info = self.client.datacenters.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)