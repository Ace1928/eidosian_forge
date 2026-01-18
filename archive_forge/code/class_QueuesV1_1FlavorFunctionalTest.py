import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1_1FlavorFunctionalTest(base.QueuesTestBase):

    def test_flavor_create(self):
        pool_data = {'uri': 'mongodb://127.0.0.1:27017', 'weight': 10, 'flavor': 'tasty'}
        pool = self.client.pool('stomach', **pool_data)
        self.addCleanup(pool.delete)
        pool_list = ['stomach']
        flavor_data = {'pool_list': pool_list}
        flavor = self.client.flavor('tasty', **flavor_data)
        self.addCleanup(flavor.delete)
        self.assertEqual('tasty', flavor.name)
        self.assertEqual(pool_list, flavor.pool_list)

    def test_flavor_get(self):
        pool_data = {'weight': 10, 'flavor': 'tasty', 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('stomach', **pool_data)
        self.addCleanup(pool.delete)
        pool_list = ['stomach']
        flavor_data = {'pool_list': pool_list}
        flavor = self.client.flavor('tasty', **flavor_data)
        resp_data = flavor.get()
        self.addCleanup(flavor.delete)
        self.assertEqual('tasty', resp_data['name'])

    def test_flavor_update(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017', 'flavor': 'tasty'}
        pool = self.client.pool('stomach', **pool_data)
        self.addCleanup(pool.delete)
        pool_list = ['stomach']
        flavor_data = {'pool_list': pool_list}
        flavor = self.client.flavor('tasty', **flavor_data)
        self.addCleanup(flavor.delete)

    def test_flavor_list(self):
        pool_data = {'uri': 'mongodb://127.0.0.1:27017', 'weight': 10, 'flavor': 'test_flavor'}
        pool = self.client.pool('stomach', **pool_data)
        self.addCleanup(pool.delete)
        pool_list = ['stomach']
        flavor_data = {'pool_list': pool_list}
        flavor = self.client.flavor('test_flavor', **flavor_data)
        self.addCleanup(flavor.delete)
        flavors = self.client.flavors()
        self.assertIsInstance(flavors, iterator._Iterator)
        self.assertEqual(1, len(list(flavors)))

    def test_flavor_delete(self):
        pool_data = {'uri': 'mongodb://127.0.0.1:27017', 'weight': 10, 'flavor': 'tasty'}
        pool = self.client.pool('stomach', **pool_data)
        self.addCleanup(pool.delete)
        pool_list = ['stomach']
        flavor_data = {'pool_list': pool_list}
        flavor = self.client.flavor('tasty', **flavor_data)
        flavor.delete()