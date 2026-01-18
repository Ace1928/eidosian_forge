from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.load_balancer_types import BoundLoadBalancerType
class AnsibleHCloudLoadBalancerTypeInfo(AnsibleHCloud):
    represent = 'hcloud_load_balancer_type_info'
    hcloud_load_balancer_type_info: list[BoundLoadBalancerType] | None = None

    def _prepare_result(self):
        tmp = []
        for load_balancer_type in self.hcloud_load_balancer_type_info:
            if load_balancer_type is not None:
                tmp.append({'id': to_native(load_balancer_type.id), 'name': to_native(load_balancer_type.name), 'description': to_native(load_balancer_type.description), 'max_connections': load_balancer_type.max_connections, 'max_services': load_balancer_type.max_services, 'max_targets': load_balancer_type.max_targets, 'max_assigned_certificates': load_balancer_type.max_assigned_certificates})
        return tmp

    def get_load_balancer_types(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_load_balancer_type_info = [self.client.load_balancer_types.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_load_balancer_type_info = [self.client.load_balancer_types.get_by_name(self.module.params.get('name'))]
            else:
                self.hcloud_load_balancer_type_info = self.client.load_balancer_types.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)