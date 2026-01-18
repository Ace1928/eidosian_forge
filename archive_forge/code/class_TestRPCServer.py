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
class TestRPCServer(test_utils.BaseTestCase, ServerSetupMixin):

    def __init__(self, *args):
        super(TestRPCServer, self).__init__(*args)
        ServerSetupMixin.__init__(self)

    def setUp(self):
        super(TestRPCServer, self).setUp(conf=cfg.ConfigOpts())
        self.useFixture(fixtures.MonkeyPatch('oslo_messaging._drivers.impl_fake.FakeExchangeManager._exchanges', new_value={}))

    def test_constructor(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(topic='foo', server='bar')
        endpoints = [object()]
        serializer = object()
        access_policy = dispatcher.DefaultRPCAccessPolicy
        server = oslo_messaging.get_rpc_server(transport, target, endpoints, serializer=serializer, access_policy=access_policy, executor='threading')
        self.assertIs(server.conf, self.conf)
        self.assertIs(server.transport, transport)
        self.assertIsInstance(server.dispatcher, oslo_messaging.RPCDispatcher)
        self.assertIs(server.dispatcher.endpoints, endpoints)
        self.assertIs(server.dispatcher.serializer, serializer)
        self.assertEqual('threading', server.executor_type)

    def test_constructor_with_eventlet_executor(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(topic='foo', server='bar')
        endpoints = [object()]
        serializer = object()
        access_policy = dispatcher.DefaultRPCAccessPolicy
        server = oslo_messaging.get_rpc_server(transport, target, endpoints, serializer=serializer, access_policy=access_policy, executor='eventlet')
        self.assertIs(server.conf, self.conf)
        self.assertIs(server.transport, transport)
        self.assertIsInstance(server.dispatcher, oslo_messaging.RPCDispatcher)
        self.assertIs(server.dispatcher.endpoints, endpoints)
        self.assertIs(server.dispatcher.serializer, serializer)
        self.assertEqual('eventlet', server.executor_type)

    def test_constructor_with_unrecognized_executor(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(topic='foo', server='bar')
        endpoints = [object()]
        serializer = object()
        access_policy = dispatcher.DefaultRPCAccessPolicy
        self.assertRaises(server_module.ExecutorLoadFailure, oslo_messaging.get_rpc_server, transport=transport, target=target, endpoints=endpoints, serializer=serializer, access_policy=access_policy, executor='boom')

    def test_server_wait_method(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(topic='foo', server='bar')
        endpoints = [object()]
        serializer = object()

        class MagicMockIgnoreArgs(mock.MagicMock):
            """MagicMock ignores arguments.

            A MagicMock which can never misinterpret the arguments passed to
            it during construction.
            """

            def __init__(self, *args, **kwargs):
                super(MagicMockIgnoreArgs, self).__init__()
        server = oslo_messaging.get_rpc_server(transport, target, endpoints, serializer=serializer)
        server._executor_cls = MagicMockIgnoreArgs
        server._create_listener = MagicMockIgnoreArgs()
        server.dispatcher = MagicMockIgnoreArgs()
        server.start()
        listener = server.listener
        server.stop()
        server.wait()
        self.assertEqual(1, listener.cleanup.call_count)

    def test_no_target_server(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        server = oslo_messaging.get_rpc_server(transport, oslo_messaging.Target(topic='testtopic'), [])
        try:
            server.start()
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.InvalidTarget, ex)
            self.assertEqual('testtopic', ex.target.topic)
        else:
            self.assertTrue(False)

    def test_no_server_topic(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(server='testserver')
        server = oslo_messaging.get_rpc_server(transport, target, [])
        try:
            server.start()
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.InvalidTarget, ex)
            self.assertEqual('testserver', ex.target.server)
        else:
            self.assertTrue(False)

    def _test_no_client_topic(self, call=True):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        client = self._setup_client(transport, topic=None)
        method = client.call if call else client.cast
        try:
            method({}, 'ping', arg='foo')
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.InvalidTarget, ex)
            self.assertIsNotNone(ex.target)
        else:
            self.assertTrue(False)

    def test_no_client_topic_call(self):
        self._test_no_client_topic(call=True)

    def test_no_client_topic_cast(self):
        self._test_no_client_topic(call=False)

    def test_client_call_timeout(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        finished = False
        wait = threading.Condition()

        class TestEndpoint(object):

            def ping(self, ctxt, arg):
                with wait:
                    if not finished:
                        wait.wait()
        server_thread = self._setup_server(transport, TestEndpoint())
        client = self._setup_client(transport)
        try:
            client.prepare(timeout=0).call({}, 'ping', arg='foo')
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.MessagingTimeout, ex)
        else:
            self.assertTrue(False)
        with wait:
            finished = True
            wait.notify()
        self._stop_server(client, server_thread)

    def test_unknown_executor(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        try:
            oslo_messaging.get_rpc_server(transport, None, [], executor='foo')
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.ExecutorLoadFailure)
            self.assertEqual('foo', ex.executor)
        else:
            self.assertTrue(False)

    def test_cast(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')

        class TestEndpoint(object):

            def __init__(self):
                self.pings = []

            def ping(self, ctxt, arg):
                self.pings.append(arg)
        endpoint = TestEndpoint()
        server_thread = self._setup_server(transport, endpoint)
        client = self._setup_client(transport)
        client.cast({}, 'ping', arg='foo')
        client.cast({}, 'ping', arg='bar')
        self._stop_server(client, server_thread)
        self.assertEqual(['dsfoo', 'dsbar'], endpoint.pings)

    def test_call(self):
        transport_srv = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        transport_cli = oslo_messaging.get_rpc_transport(self.conf, url='fake:')

        class TestEndpoint(object):

            def ping(self, ctxt, arg):
                return arg
        server_thread = self._setup_server(transport_srv, TestEndpoint())
        client = self._setup_client(transport_cli)
        self.assertIsNone(client.call({}, 'ping', arg=None))
        self.assertEqual(0, client.call({}, 'ping', arg=0))
        self.assertFalse(client.call({}, 'ping', arg=False))
        self.assertEqual([], client.call({}, 'ping', arg=[]))
        self.assertEqual({}, client.call({}, 'ping', arg={}))
        self.assertEqual('dsdsfoo', client.call({}, 'ping', arg='foo'))
        self._stop_server(client, server_thread)

    def test_direct_call(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')

        class TestEndpoint(object):

            def ping(self, ctxt, arg):
                return arg
        server_thread = self._setup_server(transport, TestEndpoint())
        client = self._setup_client(transport)
        direct = client.prepare(server='testserver')
        self.assertIsNone(direct.call({}, 'ping', arg=None))
        self.assertEqual(0, client.call({}, 'ping', arg=0))
        self.assertFalse(client.call({}, 'ping', arg=False))
        self.assertEqual([], client.call({}, 'ping', arg=[]))
        self.assertEqual({}, client.call({}, 'ping', arg={}))
        self.assertEqual('dsdsfoo', direct.call({}, 'ping', arg='foo'))
        self._stop_server(client, server_thread)

    def test_context(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')

        class TestEndpoint(object):

            def ctxt_check(self, ctxt, key):
                return ctxt[key]
        server_thread = self._setup_server(transport, TestEndpoint())
        client = self._setup_client(transport)
        self.assertEqual('dsdsb', client.call({'dsa': 'b'}, 'ctxt_check', key='a'))
        self._stop_server(client, server_thread)

    def test_failure(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')

        class TestEndpoint(object):

            def ping(self, ctxt, arg):
                raise ValueError(arg)
        debugs = []
        errors = []

        def stub_debug(msg, *a, **kw):
            if a and len(a) == 1 and isinstance(a[0], dict) and a[0]:
                a = a[0]
            debugs.append(str(msg) % a)

        def stub_error(msg, *a, **kw):
            if a and len(a) == 1 and isinstance(a[0], dict) and a[0]:
                a = a[0]
            errors.append(str(msg) % a)
        self.useFixture(fixtures.MockPatchObject(rpc_server_module.LOG, 'debug', stub_debug))
        self.useFixture(fixtures.MockPatchObject(rpc_server_module.LOG, 'error', stub_error))
        server_thread = self._setup_server(transport, TestEndpoint())
        client = self._setup_client(transport)
        try:
            client.call({}, 'ping', arg='foo')
        except Exception as ex:
            self.assertIsInstance(ex, ValueError)
            self.assertEqual('dsfoo', str(ex))
            self.assertTrue(len(debugs) == 0)
            self.assertGreater(len(errors), 0)
        else:
            self.assertTrue(False)
        self._stop_server(client, server_thread)

    def test_expected_failure(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        debugs = []
        errors = []

        def stub_debug(msg, *a, **kw):
            if a and len(a) == 1 and isinstance(a[0], dict) and a[0]:
                a = a[0]
            debugs.append(str(msg) % a)

        def stub_error(msg, *a, **kw):
            if a and len(a) == 1 and isinstance(a[0], dict) and a[0]:
                a = a[0]
            errors.append(str(msg) % a)
        self.useFixture(fixtures.MockPatchObject(rpc_server_module.LOG, 'debug', stub_debug))
        self.useFixture(fixtures.MockPatchObject(rpc_server_module.LOG, 'error', stub_error))

        class TestEndpoint(object):

            @oslo_messaging.expected_exceptions(ValueError)
            def ping(self, ctxt, arg):
                raise ValueError(arg)
        server_thread = self._setup_server(transport, TestEndpoint())
        client = self._setup_client(transport)
        try:
            client.call({}, 'ping', arg='foo')
        except Exception as ex:
            self.assertIsInstance(ex, ValueError)
            self.assertEqual('dsfoo', str(ex))
            self.assertGreater(len(debugs), 0)
            self.assertTrue(len(errors) == 0)
        else:
            self.assertTrue(False)
        self._stop_server(client, server_thread)

    @mock.patch('oslo_messaging.rpc.server.LOG')
    def test_warning_when_notifier_transport(self, log):
        transport = oslo_messaging.get_notification_transport(self.conf)
        target = oslo_messaging.Target(topic='foo', server='bar')
        endpoints = [object()]
        serializer = object()
        oslo_messaging.get_rpc_server(transport, target, endpoints, serializer=serializer)
        log.warning.assert_called_once_with('Using notification transport for RPC. Please use get_rpc_transport to obtain an RPC transport instance.')