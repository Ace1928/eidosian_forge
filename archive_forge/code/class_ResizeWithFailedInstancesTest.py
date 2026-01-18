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
class ResizeWithFailedInstancesTest(InstanceGroupWithNestedStack):
    scenarios = [('1', dict(size=3, failed=['r1'], content={'r2', 'r3', 'r4'})), ('2', dict(size=3, failed=['r4'], content={'r1', 'r2', 'r3'})), ('3', dict(size=2, failed=['r1', 'r2'], content={'r3', 'r4'})), ('4', dict(size=2, failed=['r3', 'r4'], content={'r1', 'r2'})), ('5', dict(size=2, failed=['r2', 'r3'], content={'r1', 'r4'})), ('6', dict(size=3, failed=['r2', 'r3'], content={'r1', 'r3', 'r4'}))]

    def setUp(self):
        super(ResizeWithFailedInstancesTest, self).setUp()
        nested = self.get_fake_nested_stack(4)
        inspector = mock.Mock(spec=grouputils.GroupInspector)
        self.patchobject(grouputils.GroupInspector, 'from_parent_resource', return_value=inspector)
        inspector.member_names.return_value = self.failed + sorted(self.content - set(self.failed))
        inspector.template.return_value = nested.defn._template

    def test_resize(self):
        self.group.resize(self.size)
        tmpl = self.group.update_with_template.call_args[0][0]
        resources = tmpl.resource_definitions(None)
        self.assertEqual(self.content, set(resources.keys()))