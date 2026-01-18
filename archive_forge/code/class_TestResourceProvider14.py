import operator
import uuid
from osc_placement.tests.functional import base
class TestResourceProvider14(base.BaseTestCase):
    VERSION = '1.4'

    def test_return_empty_list_for_nonexistent_aggregate(self):
        self.resource_provider_create()
        agg = str(uuid.uuid4())
        rps, warning = self.resource_provider_list(aggregate_uuids=[agg], may_print_to_stderr=True)
        self.assertEqual([], rps)
        self.assertIn('The --aggregate-uuid option is deprecated, please use --member-of instead.', warning)
        rps = self.resource_provider_list(member_of=[agg])
        self.assertEqual([], rps)

    def test_return_properly_for_aggregate_uuid_request(self):
        self.resource_provider_create()
        rp2 = self.resource_provider_create()
        agg = str(uuid.uuid4())
        self.resource_provider_aggregate_set(rp2['uuid'], agg)
        rps, warning = self.resource_provider_list(aggregate_uuids=[agg, str(uuid.uuid4())], may_print_to_stderr=True)
        self.assertEqual(1, len(rps))
        self.assertEqual(rp2['uuid'], rps[0]['uuid'])
        self.assertIn('The --aggregate-uuid option is deprecated, please use --member-of instead.', warning)
        rps = self.resource_provider_list(member_of=[agg])
        self.assertEqual(1, len(rps))
        self.assertEqual(rp2['uuid'], rps[0]['uuid'])

    def test_return_empty_list_if_no_resource(self):
        rp = self.resource_provider_create()
        self.assertEqual([], self.resource_provider_list(resources=['MEMORY_MB=256'], uuid=rp['uuid']))

    def test_return_properly_for_resource_request(self):
        rp1 = self.resource_provider_create()
        rp2 = self.resource_provider_create()
        self.resource_inventory_set(rp1['uuid'], 'PCI_DEVICE=8')
        self.resource_inventory_set(rp2['uuid'], 'PCI_DEVICE=16')
        rps = self.resource_provider_list(resources=['PCI_DEVICE=16'])
        self.assertEqual(1, len(rps))
        self.assertEqual(rp2['uuid'], rps[0]['uuid'])