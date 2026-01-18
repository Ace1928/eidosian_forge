import json
from unittest import mock
from zaqarclient.queues.v1 import iterator
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
class QueuesV1PoolUnitTest(base.QueuesTestBase):

    def test_pool_create(self):
        pool_data = {'weight': 10, 'uri': 'sqlite://'}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            send_method.return_value = resp
            pool = self.client.pool('test', **pool_data)
            self.assertEqual('test', pool.name)
            self.assertEqual(10, pool.weight)

    def test_pool_get(self):
        pool_data = {'weight': 10, 'name': 'test', 'uri': 'sqlite://', 'options': {}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(pool_data))
            send_method.return_value = resp
            pool = self.client.pool('test')
            pool1 = pool.get()
            self.assertEqual('test', pool1['name'])
            self.assertEqual(10, pool1['weight'])

    def test_pool_update(self):
        pool_data = {'weight': 10, 'uri': 'sqlite://', 'options': {}}
        updated_data = {'weight': 20, 'uri': 'sqlite://', 'options': {}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(updated_data))
            send_method.return_value = resp
            pool = self.client.pool('test', **pool_data)
            pool.update({'weight': 20})
            self.assertEqual(20, pool.weight)

    def test_pool_list(self):
        returned = {'links': [{'rel': 'next', 'href': '/v1.1/pools?marker=6244-244224-783'}], 'pools': [{'name': 'stomach', 'weight': 20, 'uri': 'sqlite://', 'options': {}}]}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, json.dumps(returned))
            send_method.return_value = resp
            pools_var = self.client.pools(limit=1)
            self.assertIsInstance(pools_var, iterator._Iterator)
            self.assertEqual(1, len(list(pools_var)))

    def test_pool_delete(self):
        pool_data = {'weight': 10, 'uri': 'sqlite://', 'options': {}}
        with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
            resp = response.Response(None, None)
            resp_data = response.Response(None, json.dumps(pool_data))
            send_method.side_effect = iter([resp_data, resp])
            pool = self.client.pool('test', **pool_data)
            pool.delete()