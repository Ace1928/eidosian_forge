from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.locations import BoundLocation
class AnsibleHCloudLocationInfo(AnsibleHCloud):
    represent = 'hcloud_location_info'
    hcloud_location_info: list[BoundLocation] | None = None

    def _prepare_result(self):
        tmp = []
        for location in self.hcloud_location_info:
            if location is not None:
                tmp.append({'id': to_native(location.id), 'name': to_native(location.name), 'description': to_native(location.description), 'city': to_native(location.city), 'country': to_native(location.country)})
        return tmp

    def get_locations(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_location_info = [self.client.locations.get_by_id(self.module.params.get('id'))]
            elif self.module.params.get('name') is not None:
                self.hcloud_location_info = [self.client.locations.get_by_name(self.module.params.get('name'))]
            else:
                self.hcloud_location_info = self.client.locations.get_all()
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, **super().base_module_arguments()), supports_check_mode=True)