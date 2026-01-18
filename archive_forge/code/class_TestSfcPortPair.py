from openstack.network.v2 import sfc_port_pair
from openstack.tests.unit import base
class TestSfcPortPair(base.TestCase):

    def test_basic(self):
        sot = sfc_port_pair.SfcPortPair()
        self.assertEqual('port_pair', sot.resource_key)
        self.assertEqual('port_pairs', sot.resources_key)
        self.assertEqual('/sfc/port_pairs', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)

    def test_make_it(self):
        sot = sfc_port_pair.SfcPortPair(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['egress'], sot.egress)
        self.assertEqual(EXAMPLE['ingress'], sot.ingress)
        self.assertEqual(EXAMPLE['service_function_parameters'], sot.service_function_parameters)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['project_id'], sot.project_id)
        self.assertDictEqual({'limit': 'limit', 'marker': 'marker', 'description': 'description', 'name': 'name', 'project_id': 'project_id', 'tenant_id': 'tenant_id', 'ingress': 'ingress', 'egress': 'egress'}, sot._query_mapping._mapping)