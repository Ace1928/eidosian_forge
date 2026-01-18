import copy
from unittest import mock
import testtools
from ironicclient import exc
from ironicclient.tests.unit import utils
import ironicclient.v1.allocation
class AllocationManagerTest(testtools.TestCase):

    def setUp(self):
        super(AllocationManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = ironicclient.v1.allocation.AllocationManager(self.api)

    def test_allocations_list(self):
        allocations = self.mgr.list()
        expect = [('GET', '/v1/allocations', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(2, len(allocations))
        expected_resp = ({}, {'allocations': [ALLOCATION, ALLOCATION2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])

    def test_allocations_list_by_node(self):
        allocations = self.mgr.list(node=ALLOCATION['node_uuid'])
        expect = [('GET', '/v1/allocations/?node=%s' % ALLOCATION['node_uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(allocations))
        expected_resp = ({}, {'allocations': [ALLOCATION, ALLOCATION2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])

    def test_allocations_list_by_owner(self):
        allocations = self.mgr.list(owner=ALLOCATION2['owner'])
        expect = [('GET', '/v1/allocations/?owner=%s' % ALLOCATION2['owner'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(1, len(allocations))
        expected_resp = ({}, {'allocations': [ALLOCATION, ALLOCATION2]})
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations']['GET'])

    def test_allocations_show(self):
        allocation = self.mgr.get(ALLOCATION['uuid'])
        expect = [('GET', '/v1/allocations/%s' % ALLOCATION['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(ALLOCATION['uuid'], allocation.uuid)
        self.assertEqual(ALLOCATION['name'], allocation.name)
        self.assertEqual(ALLOCATION['owner'], allocation.owner)
        self.assertEqual(ALLOCATION['node_uuid'], allocation.node_uuid)
        self.assertEqual(ALLOCATION['state'], allocation.state)
        self.assertEqual(ALLOCATION['resource_class'], allocation.resource_class)
        expected_resp = ({}, ALLOCATION)
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations/%s' % ALLOCATION['uuid']]['GET'])

    def test_create(self):
        allocation = self.mgr.create(**CREATE_ALLOCATION)
        expect = [('POST', '/v1/allocations', {}, CREATE_ALLOCATION)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(allocation)
        self.assertIn(ALLOCATION, self.api.responses['/v1/allocations']['GET'][1]['allocations'])

    def test_delete(self):
        allocation = self.mgr.delete(allocation_id=ALLOCATION['uuid'])
        expect = [('DELETE', '/v1/allocations/%s' % ALLOCATION['uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(allocation)
        expected_resp = ({}, ALLOCATION)
        self.assertEqual(expected_resp, self.api.responses['/v1/allocations/%s' % ALLOCATION['uuid']]['GET'])