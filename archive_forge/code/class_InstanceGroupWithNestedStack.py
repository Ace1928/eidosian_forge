import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class InstanceGroupWithNestedStack(common.HeatTestCase):

    def setUp(self):
        super(InstanceGroupWithNestedStack, self).setUp()
        t = template_format.parse(inline_templates.as_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.create_launch_config(t, self.stack)
        wsg_props = self.stack['WebServerGroup'].t._properties
        self.defn = rsrc_defn.ResourceDefinition('asg', 'OS::Heat::InstanceGroup', {'Size': 2, 'AvailabilityZones': ['zoneb'], 'LaunchConfigurationName': wsg_props['LaunchConfigurationName']})
        self.group = instgrp.InstanceGroup('asg', self.defn, self.stack)
        self.group._lb_reload = mock.Mock()
        self.group.update_with_template = mock.Mock()
        self.group.check_update_complete = mock.Mock()

    def create_launch_config(self, t, stack):
        self.stub_ImageConstraint_validate()
        self.stub_FlavorConstraint_validate()
        self.stub_SnapshotConstraint_validate()
        rsrc = stack['LaunchConfig']
        self.assertIsNone(rsrc.validate())
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        return rsrc

    def get_fake_nested_stack(self, size=1):
        tmpl = '\n        heat_template_version: 2013-05-23\n        description: AutoScaling Test\n        resources:\n        '
        resource = '\n          r%(i)d:\n            type: ResourceWithPropsAndAttrs\n            properties:\n              Foo: bar%(i)d\n          '
        resources = '\n'.join([resource % {'i': i + 1} for i in range(size)])
        nested_t = tmpl + resources
        return utils.parse_stack(template_format.parse(nested_t))