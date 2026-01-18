import copy
import json
from heatclient import exc
from oslo_log import log as logging
from testtools import matchers
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class AutoscalingGroupBasicTest(AutoscalingGroupTest):

    def test_basic_create_works(self):
        """Make sure the working case is good.

        Note this combines test_override_aws_ec2_instance into this test as
        well, which is:
        If AWS::EC2::Instance is overridden, AutoScalingGroup will
        automatically use that overridden resource type.
        """
        files = {'provider.yaml': self.instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 4, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
        initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'AWS::AutoScaling::AutoScalingGroup'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        stack = self.client.stacks.get(stack_identifier)
        self.assert_instance_count(stack, 4)

    def test_size_updates_work(self):
        files = {'provider.yaml': self.instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
        stack = self.client.stacks.get(stack_identifier)
        self.assert_instance_count(stack, 2)
        env2 = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 5, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        self.update_stack(stack_identifier, self.template, environment=env2, files=files)
        stack = self.client.stacks.get(stack_identifier)
        self.assert_instance_count(stack, 5)

    def test_update_group_replace(self):
        """Test case for ensuring non-updatable props case a replacement.

        Make sure that during a group update the non-updatable
        properties cause a replacement.
        """
        files = {'provider.yaml': self.instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': '1', 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
        rsrc = self.client.resources.get(stack_identifier, 'JobServerGroup')
        orig_asg_id = rsrc.physical_resource_id
        env2 = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': '1', 'AZ': 'wibble', 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type, 'user_data': 'new data'}}
        self.update_stack(stack_identifier, self.template, environment=env2, files=files)
        rsrc = self.client.resources.get(stack_identifier, 'JobServerGroup')
        self.assertNotEqual(orig_asg_id, rsrc.physical_resource_id)

    def test_create_instance_error_causes_group_error(self):
        """Test create failing a resource in the instance group.

        If a resource in an instance group fails to be created, the instance
        group itself will fail and the broken inner resource will remain.
        """
        stack_name = self._stack_rand_name()
        files = {'provider.yaml': self.bad_instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        self.client.stacks.create(stack_name=stack_name, template=self.template, files=files, disable_rollback=True, parameters={}, environment=env)
        self.addCleanup(self._stack_delete, stack_name)
        stack = self.client.stacks.get(stack_name)
        stack_identifier = '%s/%s' % (stack_name, stack.id)
        self._wait_for_stack_status(stack_identifier, 'CREATE_FAILED')
        initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'AWS::AutoScaling::AutoScalingGroup'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self._assert_instance_state(nested_ident, 0, 2)

    def test_update_instance_error_causes_group_error(self):
        """Test update failing a resource in the instance group.

        If a resource in an instance group fails to be created during an
        update, the instance group itself will fail and the broken inner
        resource will remain.
        """
        files = {'provider.yaml': self.instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 2, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
        initial_resources = {'JobServerConfig': 'AWS::AutoScaling::LaunchConfiguration', 'JobServerGroup': 'AWS::AutoScaling::AutoScalingGroup'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        stack = self.client.stacks.get(stack_identifier)
        self.assert_instance_count(stack, 2)
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self._assert_instance_state(nested_ident, 2, 0)
        initial_list = [res.resource_name for res in self.client.resources.list(nested_ident)]
        env['parameters']['size'] = 3
        files2 = {'provider.yaml': self.bad_instance_template}
        self.client.stacks.update(stack_id=stack_identifier, template=self.template, files=files2, disable_rollback=True, parameters={}, environment=env)
        self._wait_for_stack_status(stack_identifier, 'UPDATE_FAILED')
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        for res in self.client.resources.list(nested_ident):
            if res.resource_name in initial_list:
                self._wait_for_resource_status(nested_ident, res.resource_name, 'UPDATE_FAILED')
            else:
                self._wait_for_resource_status(nested_ident, res.resource_name, 'CREATE_FAILED')

    def test_group_suspend_resume(self):
        files = {'provider.yaml': self.instance_template}
        env = {'resource_registry': {'AWS::EC2::Instance': 'provider.yaml'}, 'parameters': {'size': 4, 'image': self.conf.minimal_image_ref, 'flavor': self.conf.instance_type}}
        stack_identifier = self.stack_create(template=self.template, files=files, environment=env)
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'JobServerGroup')
        self.stack_suspend(stack_identifier)
        self._wait_for_all_resource_status(nested_ident, 'SUSPEND_COMPLETE')
        self.stack_resume(stack_identifier)
        self._wait_for_all_resource_status(nested_ident, 'RESUME_COMPLETE')