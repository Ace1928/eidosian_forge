from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.primary_ips import BoundPrimaryIP
class AnsibleHCloudPrimaryIPInfo(AnsibleHCloud):
    represent = 'hcloud_primary_ip_info'
    hcloud_primary_ip_info: list[BoundPrimaryIP] | None = None

    def _prepare_result(self):
        tmp = []
        for primary_ip in self.hcloud_primary_ip_info:
            if primary_ip is not None:
                dns_ptr = None
                if len(primary_ip.dns_ptr) > 0:
                    dns_ptr = primary_ip.dns_ptr[0]['dns_ptr']
                tmp.append({'id': to_native(primary_ip.id), 'name': to_native(primary_ip.name), 'ip': to_native(primary_ip.ip), 'type': to_native(primary_ip.type), 'assignee_id': to_native(primary_ip.assignee_id) if primary_ip.assignee_id is not None else None, 'assignee_type': to_native(primary_ip.assignee_type), 'home_location': to_native(primary_ip.datacenter.name), 'dns_ptr': to_native(dns_ptr) if dns_ptr is not None else None, 'labels': primary_ip.labels, 'delete_protection': primary_ip.protection['delete']})
        return tmp

    def get_primary_ips(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_primary_ip_info = [self.client.primary_ips.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_primary_ip_info = [self.client.primary_ips.get_by_name(self.module.params.get('name'))]
            elif self.module.params.get('label_selector') is not None:
                self.hcloud_primary_ip_info = self.client.primary_ips.get_all(label_selector=self.module.params.get('label_selector'))
            else:
                self.hcloud_primary_ip_info = self.client.primary_ips.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, label_selector={'type': 'str'}, name={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)