import os
import queue
import time
import uuid
import fixtures
from oslo_config import cfg
import oslo_messaging
from oslo_messaging._drivers.kafka_driver import kafka_options
from oslo_messaging.notify import notifier
from oslo_messaging.tests import utils as test_utils
class RpcServerGroupFixture(fixtures.Fixture):

    def __init__(self, conf, url, topic=None, names=None, exchange=None, use_fanout_ctrl=False, endpoint=None):
        self.conf = conf
        self.url = url
        self.topic = topic or str(uuid.uuid4())
        self.names = names or ['server_%i_%s' % (i, str(uuid.uuid4())[:8]) for i in range(3)]
        self.exchange = exchange
        self.targets = [self._target(server=n) for n in self.names]
        self.use_fanout_ctrl = use_fanout_ctrl
        self.endpoint = endpoint

    def setUp(self):
        super(RpcServerGroupFixture, self).setUp()
        self.servers = [self.useFixture(self._server(t)) for t in self.targets]

    def _target(self, server=None, fanout=False):
        t = oslo_messaging.Target(exchange=self.exchange, topic=self.topic)
        t.server = server
        t.fanout = fanout
        return t

    def _server(self, target):
        ctrl = None
        if self.use_fanout_ctrl:
            ctrl = self._target(fanout=True)
        server = RpcServerFixture(self.conf, self.url, target, endpoint=self.endpoint, ctrl_target=ctrl)
        return server

    def client(self, server=None, cast=False):
        if server is None:
            target = self._target()
        elif server == 'all':
            target = self._target(fanout=True)
        elif 0 <= server < len(self.targets):
            target = self.targets[server]
        else:
            raise ValueError('Invalid value for server: %r' % server)
        transport = self.useFixture(RPCTransportFixture(self.conf, self.url))
        client = ClientStub(transport.transport, target, cast=cast, timeout=5)
        transport.wait()
        return client

    def sync(self, server=None):
        if server is None:
            for i in range(len(self.servers)):
                self.client(i).ping()
        elif server == 'all':
            for s in self.servers:
                s.syncq.get(timeout=5)
        elif 0 <= server < len(self.targets):
            self.servers[server].syncq.get(timeout=5)
        else:
            raise ValueError('Invalid value for server: %r' % server)