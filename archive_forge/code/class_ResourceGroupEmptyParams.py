import copy
from unittest import mock
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_group
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
class ResourceGroupEmptyParams(common.HeatTestCase):
    """This class tests ResourceGroup.build_resource_definition()."""
    scenarios = [('non_empty', dict(value='Bar', expected={'Foo': 'Bar'}, expected_include={'Foo': 'Bar'})), ('empty_None', dict(value=None, expected={}, expected_include={'Foo': None})), ('empty_boolean', dict(value=False, expected={'Foo': False}, expected_include={'Foo': False})), ('empty_string', dict(value='', expected={'Foo': ''}, expected_include={'Foo': ''})), ('empty_number', dict(value=0, expected={'Foo': 0}, expected_include={'Foo': 0})), ('empty_json', dict(value={}, expected={'Foo': {}}, expected_include={'Foo': {}})), ('empty_list', dict(value=[], expected={'Foo': []}, expected_include={'Foo': []}))]

    def test_definition(self):
        templ = copy.deepcopy(template)
        res_def = templ['resources']['group1']['properties']['resource_def']
        res_def['properties']['Foo'] = self.value
        stack = utils.parse_stack(templ)
        snip = stack.t.resource_definitions(stack)['group1']
        resg = resource_group.ResourceGroup('test', snip, stack)
        exp1 = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', self.expected)
        exp2 = rsrc_defn.ResourceDefinition(None, 'OverwrittenFnGetRefIdType', self.expected_include)
        rdef = resg.get_resource_def()
        self.assertEqual(exp1, resg.build_resource_definition('0', rdef))
        rdef = resg.get_resource_def(include_all=True)
        self.assertEqual(exp2, resg.build_resource_definition('0', rdef))