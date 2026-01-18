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
class CallTestCase(utils.SkipIfNoTransportURL):

    def setUp(self):
        super(CallTestCase, self).setUp(conf=cfg.ConfigOpts())
        if self.rpc_url.startswith('kafka://'):
            self.skipTest('kafka does not support RPC API')
        self.conf.prog = 'test_prog'
        self.conf.project = 'test_project'
        self.config(heartbeat_timeout_threshold=0, group='oslo_messaging_rabbit')

    def test_specific_server(self):
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client(1)
        client.append(text='open')
        self.assertEqual('openstack', client.append(text='stack'))
        client.add(increment=2)
        self.assertEqual(12, client.add(increment=10))
        self.assertEqual(9, client.subtract(increment=3))
        self.assertEqual('openstack', group.servers[1].endpoint.sval)
        self.assertEqual(9, group.servers[1].endpoint.ival)
        for i in [0, 2]:
            self.assertEqual('', group.servers[i].endpoint.sval)
            self.assertEqual(0, group.servers[i].endpoint.ival)

    def test_server_in_group(self):
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client()
        data = [c for c in 'abcdefghijklmn']
        for i in data:
            client.append(text=i)
        for s in group.servers:
            self.assertThat(len(s.endpoint.sval), matchers.GreaterThan(0))
        actual = [[c for c in s.endpoint.sval] for s in group.servers]
        self.assertThat(actual, utils.IsValidDistributionOf(data))

    def test_different_exchanges(self):
        group1 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, use_fanout_ctrl=True))
        group2 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, exchange='a', use_fanout_ctrl=True))
        group3 = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url, exchange='b', use_fanout_ctrl=True))
        client1 = group1.client(1)
        data1 = [c for c in 'abcdefghijklmn']
        for i in data1:
            client1.append(text=i)
        client2 = group2.client()
        data2 = [c for c in 'opqrstuvwxyz']
        for i in data2:
            client2.append(text=i)
        actual1 = [[c for c in s.endpoint.sval] for s in group1.servers]
        self.assertThat(actual1, utils.IsValidDistributionOf(data1))
        actual1 = [c for c in group1.servers[1].endpoint.sval]
        self.assertThat([actual1], utils.IsValidDistributionOf(data1))
        for s in group1.servers:
            expected = len(data1) if group1.servers.index(s) == 1 else 0
            self.assertEqual(expected, len(s.endpoint.sval))
            self.assertEqual(0, s.endpoint.ival)
        actual2 = [[c for c in s.endpoint.sval] for s in group2.servers]
        for s in group2.servers:
            self.assertThat(len(s.endpoint.sval), matchers.GreaterThan(0))
            self.assertEqual(0, s.endpoint.ival)
        self.assertThat(actual2, utils.IsValidDistributionOf(data2))
        for s in group3.servers:
            self.assertEqual(0, len(s.endpoint.sval))
            self.assertEqual(0, s.endpoint.ival)

    def test_timeout(self):
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        target = oslo_messaging.Target(topic='no_such_topic')
        c = utils.ClientStub(transport.transport, target, timeout=1)
        self.assertThat(c.ping, matchers.raises(oslo_messaging.MessagingTimeout))

    def test_exception(self):
        group = self.useFixture(utils.RpcServerGroupFixture(self.conf, self.rpc_url))
        client = group.client(1)
        client.add(increment=2)
        self.assertRaises(ValueError, client.subtract, increment=3)

    def test_timeout_with_concurrently_queues(self):
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
        server = self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading'))
        client = utils.ClientStub(transport.transport, target, cast=False, timeout=5)

        def short_periodical_tasks():
            for i in range(10):
                client.add(increment=1)
                time.sleep(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(client.long_running_task, seconds=10)
            executor.submit(short_periodical_tasks)
            self.assertRaises(oslo_messaging.MessagingTimeout, future.result)
        self.assertEqual(10, server.endpoint.ival)

    def test_mandatory_call(self):
        if not self.rpc_url.startswith('rabbit://'):
            self.skipTest('backend does not support call monitoring')
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
        options = oslo_messaging.TransportOptions(at_least_once=False)
        client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=1, transport_options=options)
        self.assertRaises(oslo_messaging.MessagingTimeout, client1.delay)
        options2 = oslo_messaging.TransportOptions(at_least_once=True)
        client2 = utils.ClientStub(transport.transport, target, cast=False, timeout=60, transport_options=options2)
        self.assertRaises(oslo_messaging.MessageUndeliverable, client2.delay)

    def test_monitor_long_call(self):
        if not (self.rpc_url.startswith('rabbit://') or self.rpc_url.startswith('amqp://')):
            self.skipTest('backend does not support call monitoring')
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))

        class _endpoint(object):

            def delay(self, ctxt, seconds):
                time.sleep(seconds)
                return seconds
        self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading', endpoint=_endpoint()))
        client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=1)
        self.assertRaises(oslo_messaging.MessagingTimeout, client1.delay, seconds=4)
        client2 = utils.ClientStub(transport.transport, target, cast=False, timeout=3600, call_monitor_timeout=2)
        self.assertEqual(4, client2.delay(seconds=4))

    def test_endpoint_version_namespace(self):
        target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()), namespace='Name1', version='7.5')

        class _endpoint(object):

            def __init__(self, target):
                self.target = target()

            def test(self, ctxt, echo):
                return echo
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        self.useFixture(utils.RpcServerFixture(self.conf, self.rpc_url, target, executor='threading', endpoint=_endpoint(target)))
        client1 = utils.ClientStub(transport.transport, target, cast=False, timeout=5)
        self.assertEqual('Hi there', client1.test(echo='Hi there'))
        target2 = target()
        target2.version = '7.6'
        client2 = utils.ClientStub(transport.transport, target2, cast=False, timeout=5)
        self.assertRaises(oslo_messaging.rpc.client.RemoteError, client2.test, echo='Expect failure')
        target3 = oslo_messaging.Target(topic=target.topic, server=target.server, version=target.version, namespace='Name2')
        client3 = utils.ClientStub(transport.transport, target3, cast=False, timeout=5)
        self.assertRaises(oslo_messaging.rpc.client.RemoteError, client3.test, echo='Expect failure')

    def test_bad_endpoint(self):

        class _endpoint(object):

            def target(self, ctxt, echo):
                return echo
        target = oslo_messaging.Target(topic='topic_' + str(uuid.uuid4()), server='server_' + str(uuid.uuid4()))
        transport = self.useFixture(utils.RPCTransportFixture(self.conf, self.rpc_url))
        self.assertRaises(TypeError, oslo_messaging.get_rpc_server, transport=transport.transport, target=target, endpoints=[_endpoint()], executor='threading')