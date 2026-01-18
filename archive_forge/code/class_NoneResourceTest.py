from unittest import mock
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.openstack.heat import none_resource as none
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class NoneResourceTest(common.HeatTestCase):
    tmpl = '\nheat_template_version: 2015-10-15\nresources:\n  none:\n    type: OS::Heat::None\n    properties:\n      ignored: foo\noutputs:\n  anything:\n    value: {get_attr: [none, anything]}\n'

    def _create_none_stack(self):
        self.t = template_format.parse(self.tmpl)
        self.stack = utils.parse_stack(self.t)
        self.rsrc = self.stack['none']
        self.assertIsNone(self.rsrc.validate())
        self.stack.create()
        self.assertEqual(self.rsrc.CREATE, self.rsrc.action)
        self.assertEqual(self.rsrc.COMPLETE, self.rsrc.status)
        self.assertEqual(self.stack.CREATE, self.stack.action)
        self.assertEqual(self.stack.COMPLETE, self.stack.status)
        self.stack._update_all_resource_data(False, True)
        self.assertIsNone(self.stack.outputs['anything'].get_value())

    def test_none_stack_create(self):
        self._create_none_stack()

    def test_none_stack_update_nochange(self):
        self._create_none_stack()
        before_refid = self.rsrc.FnGetRefId()
        self.assertIsNotNone(before_refid)
        utils.update_stack(self.stack, self.t)
        self.assertEqual((self.stack.UPDATE, self.stack.COMPLETE), self.stack.state)
        self.assertEqual(before_refid, self.stack['none'].FnGetRefId())

    def test_none_stack_update_add_prop(self):
        self._create_none_stack()
        before_refid = self.rsrc.FnGetRefId()
        self.assertIsNotNone(before_refid)
        new_t = self.t.copy()
        new_t['resources']['none']['properties']['another'] = 123
        utils.update_stack(self.stack, new_t)
        self.assertEqual((self.stack.UPDATE, self.stack.COMPLETE), self.stack.state)
        self.assertEqual(before_refid, self.stack['none'].FnGetRefId())

    def test_none_stack_update_del_prop(self):
        self._create_none_stack()
        before_refid = self.rsrc.FnGetRefId()
        self.assertIsNotNone(before_refid)
        new_t = self.t.copy()
        del new_t['resources']['none']['properties']['ignored']
        utils.update_stack(self.stack, new_t)
        self.assertEqual((self.stack.UPDATE, self.stack.COMPLETE), self.stack.state)
        self.assertEqual(before_refid, self.stack['none'].FnGetRefId())