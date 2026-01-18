from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.floating_ips import BoundFloatingIP
class AnsibleHCloudFloatingIPInfo(AnsibleHCloud):
    represent = 'hcloud_floating_ip_info'
    hcloud_floating_ip_info: list[BoundFloatingIP] | None = None

    def _prepare_result(self):
        tmp = []
        for floating_ip in self.hcloud_floating_ip_info:
            if floating_ip is not None:
                server_name = None
                if floating_ip.server is not None:
                    server_name = floating_ip.server.name
                tmp.append({'id': to_native(floating_ip.id), 'name': to_native(floating_ip.name), 'description': to_native(floating_ip.description), 'ip': to_native(floating_ip.ip), 'type': to_native(floating_ip.type), 'server': to_native(server_name), 'home_location': to_native(floating_ip.home_location.name), 'labels': floating_ip.labels, 'delete_protection': floating_ip.protection['delete']})
        return tmp

    def get_floating_ips(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_floating_ip_info = [self.client.floating_ips.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_floating_ip_info = [self.client.floating_ips.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_floating_ip_info = self.client.floating_ips.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_floating_ip_info = self.client.floating_ips.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, label_selector={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)