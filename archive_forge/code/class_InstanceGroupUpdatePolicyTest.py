import copy
import json
from testtools import matchers
from heat_integrationtests.functional import functional_base
class InstanceGroupUpdatePolicyTest(InstanceGroupTest):

    def ig_tmpl_with_updt_policy(self):
        templ = json.loads(copy.deepcopy(self.template))
        up = {'RollingUpdate': {'MinInstancesInService': '1', 'MaxBatchSize': '2', 'PauseTime': 'PT1S'}}
        templ['Resources']['JobServerGroup']['UpdatePolicy'] = up
        return templ

    def update_instance_group(self, updt_template, num_updates_expected_on_updt, num_creates_expected_on_updt, num_deletes_expected_on_updt, update_replace):
        files = {'provider.yaml': self.instance_template}
        size = 5
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': size, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.minimal_instance_type}}
        stack_name = self._stack_rand_name()
        stack_identifier = self.stack_create(stack_name=stack_name, template=self.ig_tmpl_with_updt_policy(), files=files, environment=env)
        stack = self.client.stacks.get(stack_identifier)
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        conf_name = self._stack_output(stack, 'JobServerConfigRef')
        conf_name_pattern = '%s-JobServerConfig-[a-zA-Z0-9]+$' % stack_name
        self.assertThat(conf_name, matchers.MatchesRegex(conf_name_pattern))
        self.assert_instance_count(stack, size)
        init_instances = self.client.resources.list(nested_ident)
        init_names = [inst.resource_name for inst in init_instances]
        self.update_stack(stack_identifier, updt_template, environment=env, files=files)
        updt_stack = self.client.stacks.get(stack_identifier)
        updt_conf_name = self._stack_output(updt_stack, 'JobServerConfigRef')
        self.assertThat(updt_conf_name, matchers.MatchesRegex(conf_name_pattern))
        self.assertNotEqual(conf_name, updt_conf_name)
        updt_instances = self.client.resources.list(nested_ident)
        updt_names = [inst.resource_name for inst in updt_instances]
        self.assertEqual(len(init_names), len(updt_names))
        for res in updt_instances:
            self.assertEqual('UPDATE_COMPLETE', res.resource_status)
        matched_names = set(updt_names) & set(init_names)
        self.assertEqual(num_updates_expected_on_updt, len(matched_names))
        self.assertEqual(num_creates_expected_on_updt, len(set(updt_names) - set(init_names)))
        self.assertEqual(num_deletes_expected_on_updt, len(set(init_names) - set(updt_names)))
        if num_deletes_expected_on_updt > 0:
            deletes_expected = init_names[:num_deletes_expected_on_updt]
            self.assertNotIn(deletes_expected, updt_names)

    def test_instance_group_update_replace(self):
        """Test simple update replace with no conflict.

        Test simple update replace with no conflict in batch size and
        minimum instances in service.
        """
        updt_template = self.ig_tmpl_with_updt_policy()
        grp = updt_template['Resources']['JobServerGroup']
        policy = grp['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '1'
        policy['MaxBatchSize'] = '3'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['UserData'] = 'new data'
        self.update_instance_group(updt_template, num_updates_expected_on_updt=5, num_creates_expected_on_updt=0, num_deletes_expected_on_updt=0, update_replace=True)

    def test_instance_group_update_replace_with_adjusted_capacity(self):
        """Test update replace with capacity adjustment.

        Test update replace with capacity adjustment due to conflict in
        batch size and minimum instances in service.
        """
        updt_template = self.ig_tmpl_with_updt_policy()
        grp = updt_template['Resources']['JobServerGroup']
        policy = grp['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '4'
        policy['MaxBatchSize'] = '4'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['UserData'] = 'new data'
        self.update_instance_group(updt_template, num_updates_expected_on_updt=2, num_creates_expected_on_updt=3, num_deletes_expected_on_updt=3, update_replace=True)

    def test_instance_group_update_replace_huge_batch_size(self):
        """Test update replace with a huge batch size."""
        updt_template = self.ig_tmpl_with_updt_policy()
        group = updt_template['Resources']['JobServerGroup']
        policy = group['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '0'
        policy['MaxBatchSize'] = '20'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['UserData'] = 'new data'
        self.update_instance_group(updt_template, num_updates_expected_on_updt=5, num_creates_expected_on_updt=0, num_deletes_expected_on_updt=0, update_replace=True)

    def test_instance_group_update_replace_huge_min_in_service(self):
        """Update replace with huge number of minimum instances in service."""
        updt_template = self.ig_tmpl_with_updt_policy()
        group = updt_template['Resources']['JobServerGroup']
        policy = group['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '20'
        policy['MaxBatchSize'] = '2'
        policy['PauseTime'] = 'PT0S'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['UserData'] = 'new data'
        self.update_instance_group(updt_template, num_updates_expected_on_updt=3, num_creates_expected_on_updt=2, num_deletes_expected_on_updt=2, update_replace=True)

    def test_instance_group_update_no_replace(self):
        """Test simple update only and no replace with no conflict.

        Test simple update only and no replace (i.e. updated instance flavor
        in Launch Configuration) with no conflict in batch size and
        minimum instances in service.
        """
        updt_template = self.ig_tmpl_with_updt_policy()
        group = updt_template['Resources']['JobServerGroup']
        policy = group['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '1'
        policy['MaxBatchSize'] = '3'
        policy['PauseTime'] = 'PT0S'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['InstanceType'] = self.conf.instance_type
        self.update_instance_group(updt_template, num_updates_expected_on_updt=5, num_creates_expected_on_updt=0, num_deletes_expected_on_updt=0, update_replace=False)

    def test_instance_group_update_no_replace_with_adjusted_capacity(self):
        """Test update only and no replace with capacity adjustment.

        Test update only and no replace (i.e. updated instance flavor in
        Launch Configuration) with capacity adjustment due to conflict in
        batch size and minimum instances in service.
        """
        updt_template = self.ig_tmpl_with_updt_policy()
        group = updt_template['Resources']['JobServerGroup']
        policy = group['UpdatePolicy']['RollingUpdate']
        policy['MinInstancesInService'] = '4'
        policy['MaxBatchSize'] = '4'
        policy['PauseTime'] = 'PT0S'
        config = updt_template['Resources']['JobServerConfig']
        config['Properties']['InstanceType'] = self.conf.instance_type
        self.update_instance_group(updt_template, num_updates_expected_on_updt=2, num_creates_expected_on_updt=3, num_deletes_expected_on_updt=3, update_replace=False)