import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1_1PoolFunctionalTest(base.QueuesTestBase):

    def test_pool_get(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('FuncTestPool', **pool_data)
        resp_data = pool.get()
        self.addCleanup(pool.delete)
        self.assertEqual('FuncTestPool', resp_data['name'])
        self.assertEqual(10, resp_data['weight'])
        self.assertEqual('mongodb://127.0.0.1:27017', resp_data['uri'])

    def test_pool_create(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('FuncTestPool', **pool_data)
        self.addCleanup(pool.delete)
        self.assertEqual('FuncTestPool', pool.name)
        self.assertEqual(10, pool.weight)

    def test_pool_update(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('FuncTestPool', **pool_data)
        self.addCleanup(pool.delete)
        pool.update({'weight': 20})
        self.assertEqual(20, pool.weight)

    def test_pool_list(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('FuncTestPool', **pool_data)
        self.addCleanup(pool.delete)
        pools = self.client.pools()
        self.assertIsInstance(pools, iterator._Iterator)
        self.assertEqual(1, len(list(pools)))

    def test_pool_delete(self):
        pool_data = {'weight': 10, 'uri': 'mongodb://127.0.0.1:27017'}
        pool = self.client.pool('FuncTestPool', **pool_data)
        pool.delete()