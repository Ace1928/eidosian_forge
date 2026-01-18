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
class HeatScalingGroupAttrFallbackTest(common.HeatTestCase):

    def setUp(self):
        super(HeatScalingGroupAttrFallbackTest, self).setUp()
        t = template_format.parse(inline_templates.as_heat_template)
        self.stack = utils.parse_stack(t, params=inline_templates.as_params)
        self.group = self.stack['my-group']
        self.assertIsNone(self.group.validate())
        self.group.get_output = mock.Mock(side_effect=exception.NotFound)

    def test_output_attribute_list(self):
        mock_members = self.patchobject(grouputils, 'get_members')
        members = []
        output = []
        for ip_ex in range(1, 4):
            inst = mock.Mock()
            inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
            output.append('2.1.3.%d' % ip_ex)
            members.append(inst)
        mock_members.return_value = members
        self.assertEqual(output, self.group.FnGetAtt('outputs_list', 'Bar'))

    def test_output_refs(self):
        mock_get = self.patchobject(grouputils, 'get_member_refids')
        mock_get.return_value = ['resource-1', 'resource-2']
        found = self.group.FnGetAtt('refs')
        expected = ['resource-1', 'resource-2']
        self.assertEqual(expected, found)
        mock_get.assert_called_once_with(self.group)

    def test_output_refs_map(self):
        mock_members = self.patchobject(grouputils, 'get_members')
        members = [mock.MagicMock(), mock.MagicMock()]
        members[0].name = 'resource-1-name'
        members[0].resource_id = 'resource-1-id'
        members[1].name = 'resource-2-name'
        members[1].resource_id = 'resource-2-id'
        mock_members.return_value = members
        found = self.group.FnGetAtt('refs_map')
        expected = {'resource-1-name': 'resource-1-id', 'resource-2-name': 'resource-2-id'}
        self.assertEqual(expected, found)

    def test_output_attribute_dict(self):
        mock_members = self.patchobject(grouputils, 'get_members')
        members = []
        output = {}
        for ip_ex in range(1, 4):
            inst = mock.Mock()
            inst.name = str(ip_ex)
            inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
            output[str(ip_ex)] = '2.1.3.%d' % ip_ex
            members.append(inst)
        mock_members.return_value = members
        self.assertEqual(output, self.group.FnGetAtt('outputs', 'Bar'))

    def test_index_dotted_attribute(self):
        mock_members = self.patchobject(grouputils, 'get_members')
        self.group.nested = mock.Mock()
        members = []
        output = []
        for ip_ex in range(0, 2):
            inst = mock.Mock()
            inst.name = 'ab'[ip_ex]
            inst.FnGetAtt.return_value = '2.1.3.%d' % ip_ex
            output.append('2.1.3.%d' % ip_ex)
            members.append(inst)
        mock_members.return_value = members
        self.assertEqual(output[0], self.group.FnGetAtt('resource.0', 'Bar'))
        self.assertEqual(output[1], self.group.FnGetAtt('resource.1.Bar'))
        self.assertRaises(exception.NotFound, self.group.FnGetAtt, 'resource.2')