import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
class TestMultipleServers(test_utils.BaseTestCase, ServerSetupMixin):
    _exchanges = [('same_exchange', dict(exchange1=None, exchange2=None)), ('diff_exchange', dict(exchange1='x1', exchange2='x2'))]
    _topics = [('same_topic', dict(topic1='t', topic2='t')), ('diff_topic', dict(topic1='t1', topic2='t2'))]
    _server = [('same_server', dict(server1=None, server2=None)), ('diff_server', dict(server1='s1', server2='s2'))]
    _fanout = [('not_fanout', dict(fanout1=None, fanout2=None)), ('fanout', dict(fanout1=True, fanout2=True))]
    _method = [('call', dict(call1=True, call2=True)), ('cast', dict(call1=False, call2=False))]
    _endpoints = [('one_endpoint', dict(multi_endpoints=False, expect1=['ds1', 'ds2'], expect2=['ds1', 'ds2'])), ('two_endpoints', dict(multi_endpoints=True, expect1=['ds1'], expect2=['ds2']))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._exchanges, cls._topics, cls._server, cls._fanout, cls._method, cls._endpoints)

        def filter_fanout_call(scenario):
            params = scenario[1]
            fanout = params['fanout1'] or params['fanout2']
            call = params['call1'] or params['call2']
            return not (call and fanout)

        def filter_same_topic_and_server(scenario):
            params = scenario[1]
            single_topic = params['topic1'] == params['topic2']
            single_server = params['server1'] == params['server2']
            return not (single_topic and single_server)

        def fanout_to_servers(scenario):
            params = scenario[1]
            fanout = params['fanout1'] or params['fanout2']
            single_exchange = params['exchange1'] == params['exchange2']
            single_topic = params['topic1'] == params['topic2']
            multi_servers = params['server1'] != params['server2']
            if fanout and single_exchange and single_topic and multi_servers:
                params['expect1'] = params['expect1'][:] + params['expect1']
                params['expect2'] = params['expect2'][:] + params['expect2']
            return scenario

        def single_topic_multi_endpoints(scenario):
            params = scenario[1]
            single_exchange = params['exchange1'] == params['exchange2']
            single_topic = params['topic1'] == params['topic2']
            if single_topic and single_exchange and params['multi_endpoints']:
                params['expect_either'] = params['expect1'] + params['expect2']
                params['expect1'] = params['expect2'] = []
            else:
                params['expect_either'] = []
            return scenario
        for f in [filter_fanout_call, filter_same_topic_and_server]:
            cls.scenarios = [i for i in cls.scenarios if f(i)]
        for m in [fanout_to_servers, single_topic_multi_endpoints]:
            cls.scenarios = [m(i) for i in cls.scenarios]

    def __init__(self, *args):
        super(TestMultipleServers, self).__init__(*args)
        ServerSetupMixin.__init__(self)

    def setUp(self):
        super(TestMultipleServers, self).setUp(conf=cfg.ConfigOpts())
        self.useFixture(fixtures.MonkeyPatch('oslo_messaging._drivers.impl_fake.FakeExchangeManager._exchanges', new_value={}))

    def test_multiple_servers(self):
        transport1 = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        if self.exchange1 != self.exchange2:
            transport2 = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        else:
            transport2 = transport1

        class TestEndpoint(object):

            def __init__(self):
                self.pings = []

            def ping(self, ctxt, arg):
                self.pings.append(arg)

            def alive(self, ctxt):
                return 'alive'
        if self.multi_endpoints:
            endpoint1, endpoint2 = (TestEndpoint(), TestEndpoint())
        else:
            endpoint1 = endpoint2 = TestEndpoint()
        server1 = self._setup_server(transport1, endpoint1, topic=self.topic1, exchange=self.exchange1, server=self.server1)
        server2 = self._setup_server(transport2, endpoint2, topic=self.topic2, exchange=self.exchange2, server=self.server2)
        client1 = self._setup_client(transport1, topic=self.topic1, exchange=self.exchange1)
        client2 = self._setup_client(transport2, topic=self.topic2, exchange=self.exchange2)
        client1 = client1.prepare(server=self.server1)
        client2 = client2.prepare(server=self.server2)
        if self.fanout1:
            client1.call({}, 'alive')
            client1 = client1.prepare(fanout=True)
        if self.fanout2:
            client2.call({}, 'alive')
            client2 = client2.prepare(fanout=True)
        (client1.call if self.call1 else client1.cast)({}, 'ping', arg='1')
        (client2.call if self.call2 else client2.cast)({}, 'ping', arg='2')
        self._stop_server(client1.prepare(fanout=None), server1, topic=self.topic1, exchange=self.exchange1)
        self._stop_server(client2.prepare(fanout=None), server2, topic=self.topic2, exchange=self.exchange2)

        def check(pings, expect):
            self.assertEqual(len(expect), len(pings))
            for a in expect:
                self.assertIn(a, pings)
        if self.expect_either:
            check(endpoint1.pings + endpoint2.pings, self.expect_either)
        else:
            check(endpoint1.pings, self.expect1)
            check(endpoint2.pings, self.expect2)