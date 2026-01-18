import os
import requests
import subprocess
import time
import uuid
import concurrent.futures
from oslo_config import cfg
from testtools import matchers
import oslo_messaging
from oslo_messaging.tests.functional import utils
class CastTestCase(utils.SkipIfNoTransportURL):

    def setUp(self):
        super(CastTestCase, self).setUp()
        if self.rpc_url.startswith('kafka://'):
            self.skipTest('kafka does not support RPC API')

    def test_specific_server(self):
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client(1, cast=True)
        client.append(text='open')
        client.append(text='stack')
        client.add(increment=2)
        client.add(increment=10)
        time.sleep(0.3)
        client.sync()
        group.sync(1)
        self.assertIn(group.servers[1].endpoint.sval, ['openstack', 'stackopen'])
        self.assertEqual(12, group.servers[1].endpoint.ival)
        for i in [0, 2]:
            self.assertEqual('', group.servers[i].endpoint.sval)
            self.assertEqual(0, group.servers[i].endpoint.ival)

    def test_server_in_group(self):
        if self.rpc_url.startswith('amqp:'):
            self.skipTest('QPID-6307')
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client(cast=True)
        for i in range(20):
            client.add(increment=1)
        for i in range(len(group.servers)):
            client.sync()
        group.sync(server='all')
        total = 0
        for s in group.servers:
            ival = s.endpoint.ival
            self.assertThat(ival, matchers.GreaterThan(0))
            self.assertThat(ival, matchers.LessThan(20))
            total += ival
        self.assertEqual(20, total)

    def test_fanout(self):
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client('all', cast=True)
        client.append(text='open')
        client.append(text='stack')
        client.add(increment=2)
        client.add(increment=10)
        time.sleep(0.3)
        client.sync()
        group.sync(server='all')
        for s in group.servers:
            self.assertIn(s.endpoint.sval, ['openstack', 'stackopen'])
            self.assertEqual(12, s.endpoint.ival)