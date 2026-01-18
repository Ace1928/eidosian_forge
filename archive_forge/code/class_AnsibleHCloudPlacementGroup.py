from __future__ import annotations
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ..module_utils.hcloud import AnsibleHCloud
from ..module_utils.vendor.hcloud import HCloudException
from ..module_utils.vendor.hcloud.placement_groups import BoundPlacementGroup
class AnsibleHCloudPlacementGroup(AnsibleHCloud):
    represent = 'hcloud_placement_group'
    hcloud_placement_group: BoundPlacementGroup | None = None

    def _prepare_result(self):
        return {'id': to_native(self.hcloud_placement_group.id), 'name': to_native(self.hcloud_placement_group.name), 'labels': self.hcloud_placement_group.labels, 'type': to_native(self.hcloud_placement_group.type), 'servers': self.hcloud_placement_group.servers}

    def _get_placement_group(self):
        try:
            if self.module.params.get('id') is not None:
                self.hcloud_placement_group = self.client.placement_groups.get_by_id(self.module.params.get('id'))
            elif self.module.params.get('name') is not None:
                self.hcloud_placement_group = self.client.placement_groups.get_by_name(self.module.params.get('name'))
        except HCloudException as exception:
            self.fail_json_hcloud(exception)

    def _create_placement_group(self):
        self.module.fail_on_missing_params(required_params=['name'])
        params = {'name': self.module.params.get('name'), 'type': self.module.params.get('type'), 'labels': self.module.params.get('labels')}
        if not self.module.check_mode:
            try:
                self.client.placement_groups.create(**params)
            except HCloudException as exception:
                self.fail_json_hcloud(exception, params=params)
        self._mark_as_changed()
        self._get_placement_group()

    def _update_placement_group(self):
        name = self.module.params.get('name')
        if name is not None and self.hcloud_placement_group.name != name:
            self.module.fail_on_missing_params(required_params=['id'])
            if not self.module.check_mode:
                self.hcloud_placement_group.update(name=name)
            self._mark_as_changed()
        labels = self.module.params.get('labels')
        if labels is not None and self.hcloud_placement_group.labels != labels:
            if not self.module.check_mode:
                self.hcloud_placement_group.update(labels=labels)
            self._mark_as_changed()
        self._get_placement_group()

    def present_placement_group(self):
        self._get_placement_group()
        if self.hcloud_placement_group is None:
            self._create_placement_group()
        else:
            self._update_placement_group()

    def delete_placement_group(self):
        self._get_placement_group()
        if self.hcloud_placement_group is not None:
            if not self.module.check_mode:
                self.client.placement_groups.delete(self.hcloud_placement_group)
            self._mark_as_changed()
        self.hcloud_placement_group = None

    @classmethod
    def define_module(cls):
        return AnsibleModule(argument_spec=dict(id={'type': 'int'}, name={'type': 'str'}, labels={'type': 'dict'}, type={'type': 'str'}, state={'choices': ['absent', 'present'], 'default': 'present'}, **super().base_module_arguments()), required_one_of=[['id', 'name']], required_if=[['state', 'present', ['name']]], supports_check_mode=True)