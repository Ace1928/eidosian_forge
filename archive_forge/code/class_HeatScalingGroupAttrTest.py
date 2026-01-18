import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
class HeatScalingGroupAttrTest(common.HeatTestCase):

    def setUp(self):
        super(HeatScalingGroupAttrTest, self).setUp()
        t = template_format.parse(inline_templates.as_heat_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.group = self.stack['my-group']
        self.assertIsNone(self.group.validate())

    def test_no_instance_list(self):
        """Tests inheritance of InstanceList attribute.

        The InstanceList attribute is not inherited from
        AutoScalingResourceGroup's superclasses.
        """
        self.assertRaises(exception.InvalidTemplateAttribute, self.group.FnGetAtt, 'InstanceList')

    def _stub_get_attr(self, refids, attrs):

        def ref_id_fn(res_name):
            return refids[res_name]

        def attr_fn(args):
            res_name = args[0]
            return attrs[res_name]
        inspector = self.group._group_data()
        member_names = sorted(refids if refids else attrs)
        self.patchobject(inspector, 'member_names', return_value=member_names)

        def get_output(output_name):
            outputs = self.group._nested_output_defns(member_names, attr_fn, ref_id_fn)
            op_defns = {od.name: od for od in outputs}
            self.assertIn(output_name, op_defns)
            return op_defns[output_name].get_value()
        orig_get_attr = self.group.FnGetAtt

        def get_attr(attr_name, *path):
            if not path:
                attr = attr_name
            else:
                attr = (attr_name,) + path
            self.group.referenced_attrs = mock.Mock(return_value=[attr])
            return orig_get_attr(attr_name, *path)
        self.group.FnGetAtt = mock.Mock(side_effect=get_attr)
        self.group.get_output = mock.Mock(side_effect=get_output)

    def test_output_attribute_list(self):
        values = {str(i): '2.1.3.%d' % i for i in range(1, 4)}
        self._stub_get_attr({n: 'foo' for n in values}, values)
        expected = [v for k, v in sorted(values.items())]
        self.assertEqual(expected, self.group.FnGetAtt('outputs_list', 'Bar'))

    def test_output_attribute_dict(self):
        values = {str(i): '2.1.3.%d' % i for i in range(1, 4)}
        self._stub_get_attr({n: 'foo' for n in values}, values)
        self.assertEqual(values, self.group.FnGetAtt('outputs', 'Bar'))

    def test_index_dotted_attribute(self):
        values = {'ab'[i - 1]: '2.1.3.%d' % i for i in range(1, 3)}
        self._stub_get_attr({'a': 'foo', 'b': 'bar'}, values)
        self.assertEqual(values['a'], self.group.FnGetAtt('resource.0', 'Bar'))
        self.assertEqual(values['b'], self.group.FnGetAtt('resource.1.Bar'))
        self.assertRaises(exception.NotFound, self.group.FnGetAtt, 'resource.2')

    def test_output_refs(self):
        values = {'abc': 'resource-1', 'def': 'resource-2'}
        self._stub_get_attr(values, {})
        expected = [v for k, v in sorted(values.items())]
        self.assertEqual(expected, self.group.FnGetAtt('refs'))

    def test_output_refs_map(self):
        values = {'abc': 'resource-1', 'def': 'resource-2'}
        self._stub_get_attr(values, {})
        self.assertEqual(values, self.group.FnGetAtt('refs_map'))

    def test_attribute_current_size(self):
        mock_instances = self.patchobject(grouputils, 'get_size')
        mock_instances.return_value = 3
        self.assertEqual(3, self.group.FnGetAtt('current_size'))

    def test_attribute_current_size_with_path(self):
        mock_instances = self.patchobject(grouputils, 'get_size')
        mock_instances.return_value = 4
        self.assertEqual(4, self.group.FnGetAtt('current_size', 'name'))