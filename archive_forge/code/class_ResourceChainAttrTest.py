import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
class ResourceChainAttrTest(common.HeatTestCase):

    def test_aggregate_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        chain = self._create_dummy_stack()
        expected = ['0', '1']
        self.assertEqual(expected, chain.FnGetAtt('foo'))
        self.assertEqual(expected, chain.FnGetAtt('Foo'))

    def test_index_dotted_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        chain = self._create_dummy_stack()
        self.assertEqual('0', chain.FnGetAtt('resource.0.Foo'))
        self.assertEqual('1', chain.FnGetAtt('resource.1.Foo'))

    def test_index_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        chain = self._create_dummy_stack()
        self.assertEqual('0', chain.FnGetAtt('resource.0', 'Foo'))
        self.assertEqual('1', chain.FnGetAtt('resource.1', 'Foo'))

    def test_index_deep_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        chain = self._create_dummy_stack(expect_attrs={'0': 2, '1': 3})
        self.assertEqual(2, chain.FnGetAtt('resource.0', 'nested_dict', 'dict', 'b'))
        self.assertEqual(3, chain.FnGetAtt('resource.1', 'nested_dict', 'dict', 'b'))

    def test_aggregate_deep_path_attribs(self):
        """Test attribute aggregation.

        Test attribute aggregation and that we mimic the nested resource's
        attributes.
        """
        chain = self._create_dummy_stack(expect_attrs={'0': 3, '1': 3})
        expected = [3, 3]
        self.assertEqual(expected, chain.FnGetAtt('nested_dict', 'list', 2))

    def test_aggregate_refs(self):
        """Test resource id aggregation."""
        chain = self._create_dummy_stack()
        expected = ['ID-0', 'ID-1']
        self.assertEqual(expected, chain.FnGetAtt('refs'))

    def test_aggregate_refs_with_index(self):
        """Test resource id aggregation with index."""
        chain = self._create_dummy_stack()
        expected = ['ID-0', 'ID-1']
        self.assertEqual(expected[0], chain.FnGetAtt('refs', 0))
        self.assertEqual(expected[1], chain.FnGetAtt('refs', 1))
        self.assertIsNone(chain.FnGetAtt('refs', 2))

    def test_aggregate_outputs(self):
        """Test outputs aggregation."""
        expected = {'0': ['foo', 'bar'], '1': ['foo', 'bar']}
        chain = self._create_dummy_stack(expect_attrs=expected)
        self.assertEqual(expected, chain.FnGetAtt('attributes', 'list'))

    def test_aggregate_outputs_no_path(self):
        """Test outputs aggregation with missing path."""
        chain = self._create_dummy_stack()
        self.assertRaises(exception.InvalidTemplateAttribute, chain.FnGetAtt, 'attributes')

    def test_index_refs(self):
        """Tests getting ids of individual resources."""
        chain = self._create_dummy_stack()
        self.assertEqual('ID-0', chain.FnGetAtt('resource.0'))
        self.assertEqual('ID-1', chain.FnGetAtt('resource.1'))
        ex = self.assertRaises(exception.NotFound, chain.FnGetAtt, 'resource.2')
        self.assertIn("Member '2' not found in group resource 'test'", str(ex))

    def _create_dummy_stack(self, expect_count=2, expect_attrs=None):
        self.stack = utils.parse_stack(TEMPLATE)
        snip = self.stack.t.resource_definitions(self.stack)['test-chain']
        chain = resource_chain.ResourceChain('test', snip, self.stack)
        attrs = {}
        refids = {}
        if expect_attrs is None:
            expect_attrs = {}
        for index in range(expect_count):
            res = str(index)
            attrs[index] = expect_attrs.get(res, res)
            refids[index] = 'ID-%s' % res
        names = [str(name) for name in range(expect_count)]
        chain._resource_names = mock.Mock(return_value=names)
        self._stub_get_attr(chain, refids, attrs)
        return chain

    def _stub_get_attr(self, chain, refids, attrs):

        def ref_id_fn(res_name):
            return refids[int(res_name)]

        def attr_fn(args):
            res_name = args[0]
            return attrs[int(res_name)]

        def get_output(output_name):
            outputs = chain._nested_output_defns(chain._resource_names(), attr_fn, ref_id_fn)
            op_defns = {od.name: od for od in outputs}
            if output_name not in op_defns:
                raise exception.NotFound('Specified output key %s not found.' % output_name)
            return op_defns[output_name].get_value()
        orig_get_attr = chain.FnGetAtt

        def get_attr(attr_name, *path):
            if not path:
                attr = attr_name
            else:
                attr = (attr_name,) + path
            chain.referenced_attrs = mock.Mock(return_value=[attr])
            return orig_get_attr(attr_name, *path)
        chain.FnGetAtt = mock.Mock(side_effect=get_attr)
        chain.get_output = mock.Mock(side_effect=get_output)