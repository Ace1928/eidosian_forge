import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
class ResourceGroupUpdatePolicyTest(functional_base.FunctionalTestsBase):
    template = "\nheat_template_version: '2015-04-30'\nresources:\n  random_group:\n    type: OS::Heat::ResourceGroup\n    update_policy:\n      rolling_update:\n        min_in_service: 1\n        max_batch_size: 2\n        pause_time: 1\n    properties:\n      count: 10\n      resource_def:\n        type: OS::Heat::TestResource\n        properties:\n          value: initial\n          update_replace: False\n"

    def update_resource_group(self, update_template, updated, created, deleted):
        stack_identifier = self.stack_create(template=self.template)
        group_resources = self.list_group_resources(stack_identifier, 'random_group', minimal=False)
        init_names = [res.physical_resource_id for res in group_resources]
        self.update_stack(stack_identifier, update_template)
        group_resources = self.list_group_resources(stack_identifier, 'random_group', minimal=False)
        updt_names = [res.physical_resource_id for res in group_resources]
        matched_names = set(updt_names) & set(init_names)
        self.assertEqual(updated, len(matched_names))
        self.assertEqual(created, len(set(updt_names) - set(init_names)))
        self.assertEqual(deleted, len(set(init_names) - set(updt_names)))

    def test_resource_group_update(self):
        """Test rolling update with no conflict.

        Simple rolling update with no conflict in batch size
        and minimum instances in service.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '1'
        policy['max_batch_size'] = '3'
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=10, created=0, deleted=0)

    def test_resource_group_update_replace(self):
        """Test rolling update(replace)with no conflict.

        Simple rolling update replace with no conflict in batch size
        and minimum instances in service.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '1'
        policy['max_batch_size'] = '3'
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        res_def['properties']['update_replace'] = True
        self.update_resource_group(updt_template, updated=0, created=10, deleted=10)

    def test_resource_group_update_replace_template_changed(self):
        """Test rolling update(replace)with child template path changed.

        Simple rolling update replace with child template path changed.
        """
        nested_templ = '\nheat_template_version: "2013-05-23"\nresources:\n  oops:\n    type: OS::Heat::TestResource\n'
        create_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = create_template['resources']['random_group']
        grp['properties']['resource_def'] = {'type': '/opt/provider.yaml'}
        files = {'/opt/provider.yaml': nested_templ}
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '1'
        policy['max_batch_size'] = '3'
        stack_identifier = self.stack_create(template=create_template, files=files)
        update_template = create_template.copy()
        grp = update_template['resources']['random_group']
        grp['properties']['resource_def'] = {'type': '/opt1/provider.yaml'}
        files = {'/opt1/provider.yaml': nested_templ}
        self.update_stack(stack_identifier, update_template, files=files)

    def test_resource_group_update_scaledown(self):
        """Test rolling update with scaledown.

        Simple rolling update with reduced size.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '1'
        policy['max_batch_size'] = '3'
        grp['properties']['count'] = 6
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=6, created=0, deleted=4)

    def test_resource_group_update_scaleup(self):
        """Test rolling update with scaleup.

        Simple rolling update with increased size.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '1'
        policy['max_batch_size'] = '3'
        grp['properties']['count'] = 12
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=10, created=2, deleted=0)

    def test_resource_group_update_adjusted(self):
        """Test rolling update with enough available resources

        Update  with capacity adjustment with enough resources.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '8'
        policy['max_batch_size'] = '4'
        grp['properties']['count'] = 6
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=6, created=0, deleted=4)

    def test_resource_group_update_with_adjusted_capacity(self):
        """Test rolling update with capacity adjustment.

        Rolling update with capacity adjustment due to conflict in
        batch size and minimum instances in service.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '8'
        policy['max_batch_size'] = '4'
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=10, created=0, deleted=0)

    def test_resource_group_update_huge_batch_size(self):
        """Test rolling update with huge batch size.

        Rolling Update with a huge batch size(more than
        current size).
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '0'
        policy['max_batch_size'] = '20'
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=10, created=0, deleted=0)

    def test_resource_group_update_huge_min_in_service(self):
        """Test rolling update with huge minimum capacity.

        Rolling Update with a huge number of minimum instances
        in service.
        """
        updt_template = yaml.safe_load(copy.deepcopy(self.template))
        grp = updt_template['resources']['random_group']
        policy = grp['update_policy']['rolling_update']
        policy['min_in_service'] = '20'
        policy['max_batch_size'] = '1'
        res_def = grp['properties']['resource_def']
        res_def['properties']['value'] = 'updated'
        self.update_resource_group(updt_template, updated=10, created=0, deleted=0)