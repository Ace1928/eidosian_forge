import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class TestAutoScalingGroupValidation(common.HeatTestCase):

    def validate_scaling_group(self, t, stack, resource_name):
        conf = stack['LaunchConfig']
        self.assertIsNone(conf.validate())
        scheduler.TaskRunner(conf.create)()
        self.assertEqual((conf.CREATE, conf.COMPLETE), conf.state)
        rsrc = stack[resource_name]
        self.assertIsNone(rsrc.validate())
        return rsrc

    def test_toomany_vpc_zone_identifier(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['VPCZoneIdentifier'] = ['xxxx', 'yyyy']
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stub_SnapshotConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        self.assertRaises(exception.NotSupported, self.validate_scaling_group, t, stack, 'WebServerGroup')

    def test_invalid_min_size(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['MinSize'] = '-1'
        properties['MaxSize'] = '2'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stub_SnapshotConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        e = self.assertRaises(exception.StackValidationFailed, self.validate_scaling_group, t, stack, 'WebServerGroup')
        expected_msg = 'The size of AutoScalingGroup can not be less than zero'
        self.assertEqual(expected_msg, str(e))

    def test_invalid_max_size(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['MinSize'] = '3'
        properties['MaxSize'] = '1'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stub_SnapshotConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        e = self.assertRaises(exception.StackValidationFailed, self.validate_scaling_group, t, stack, 'WebServerGroup')
        expected_msg = 'MinSize can not be greater than MaxSize'
        self.assertEqual(expected_msg, str(e))

    def test_invalid_desiredcapacity(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['MinSize'] = '1'
        properties['MaxSize'] = '3'
        properties['DesiredCapacity'] = '4'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stub_SnapshotConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        e = self.assertRaises(exception.StackValidationFailed, self.validate_scaling_group, t, stack, 'WebServerGroup')
        expected_msg = 'DesiredCapacity must be between MinSize and MaxSize'
        self.assertEqual(expected_msg, str(e))

    def test_invalid_desiredcapacity_zero(self):
        t = template_format.parse(as_template)
        properties = t['Resources']['WebServerGroup']['Properties']
        properties['MinSize'] = '1'
        properties['MaxSize'] = '3'
        properties['DesiredCapacity'] = '0'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.stub_SnapshotConstraint_validate()
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        e = self.assertRaises(exception.StackValidationFailed, self.validate_scaling_group, t, stack, 'WebServerGroup')
        expected_msg = 'DesiredCapacity must be between MinSize and MaxSize'
        self.assertEqual(expected_msg, str(e))

    def test_validate_without_InstanceId_and_LaunchConfigurationName(self):
        t = template_format.parse(as_template)
        agp = t['Resources']['WebServerGroup']['Properties']
        agp.pop('LaunchConfigurationName')
        agp.pop('LoadBalancerNames')
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        rsrc = stack['WebServerGroup']
        error_msg = "Either 'InstanceId' or 'LaunchConfigurationName' must be provided."
        exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        self.assertIn(error_msg, str(exc))

    def test_validate_with_InstanceId_and_LaunchConfigurationName(self):
        t = template_format.parse(as_template)
        agp = t['Resources']['WebServerGroup']['Properties']
        agp['InstanceId'] = '5678'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        rsrc = stack['WebServerGroup']
        error_msg = "Either 'InstanceId' or 'LaunchConfigurationName' must be provided."
        exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        self.assertIn(error_msg, str(exc))

    def _stub_nova_server_get(self, not_found=False):
        mock_server = mock.MagicMock()
        mock_server.image = {'id': 'dd619705-468a-4f7d-8a06-b84794b3561a'}
        mock_server.flavor = {'id': '1'}
        mock_server.key_name = 'test'
        mock_server.security_groups = [{u'name': u'hth_test'}]
        if not_found:
            self.patchobject(nova.NovaClientPlugin, 'get_server', side_effect=exception.EntityNotFound(entity='Server', name='5678'))
        else:
            self.patchobject(nova.NovaClientPlugin, 'get_server', return_value=mock_server)

    def test_scaling_group_create_with_instanceid(self):
        t = template_format.parse(as_template)
        agp = t['Resources']['WebServerGroup']['Properties']
        agp['InstanceId'] = '5678'
        agp.pop('LaunchConfigurationName')
        agp.pop('LoadBalancerNames')
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        rsrc = stack['WebServerGroup']
        self._stub_nova_server_get()
        _config, ins_props = rsrc._get_conf_properties()
        self.assertEqual('dd619705-468a-4f7d-8a06-b84794b3561a', ins_props['ImageId'])
        self.assertEqual('test', ins_props['KeyName'])
        self.assertEqual(['hth_test'], ins_props['SecurityGroups'])
        self.assertEqual('1', ins_props['InstanceType'])

    def test_scaling_group_create_with_instanceid_not_found(self):
        t = template_format.parse(as_template)
        agp = t['Resources']['WebServerGroup']['Properties']
        agp.pop('LaunchConfigurationName')
        agp['InstanceId'] = '5678'
        stack = utils.parse_stack(t, params=inline_templates.as_params)
        rsrc = stack['WebServerGroup']
        self._stub_nova_server_get(not_found=True)
        msg = "Property error: Resources.WebServerGroup.Properties.InstanceId: Error validating value '5678': The Server (5678) could not be found."
        exc = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        self.assertIn(msg, str(exc))