from openstack.network.v2 import sfc_port_chain
from openstack.tests.unit import base
class TestPortChain(base.TestCase):

    def test_basic(self):
        sot = sfc_port_chain.SfcPortChain()
        self.assertEqual('port_chain', sot.resource_key)
        self.assertEqual('port_chains', sot.resources_key)
        self.assertEqual('/sfc/port_chains', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = sfc_port_chain.SfcPortChain(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['port_pair_groups'], sot.port_pair_groups)
        self.assertEqual(EXAMPLE['flow_classifiers'], sot.flow_classifiers)
        self.assertEqual(EXAMPLE['chain_parameters'], sot.chain_parameters)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'name': 'name', 'project_id': 'project_id', 'tenant_id': 'tenant_id'}, sot._query_mapping._mapping)