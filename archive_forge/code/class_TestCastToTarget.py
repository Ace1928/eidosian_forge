from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestCastToTarget(test_utils.BaseTestCase):
    _base = [('all_none', dict(ctor={}, prepare={}, expect={})), ('ctor_exchange', dict(ctor=dict(exchange='testexchange'), prepare={}, expect=dict(exchange='testexchange'))), ('prepare_exchange', dict(ctor={}, prepare=dict(exchange='testexchange'), expect=dict(exchange='testexchange'))), ('prepare_exchange_none', dict(ctor=dict(exchange='testexchange'), prepare=dict(exchange=None), expect={})), ('both_exchange', dict(ctor=dict(exchange='ctorexchange'), prepare=dict(exchange='testexchange'), expect=dict(exchange='testexchange'))), ('ctor_topic', dict(ctor=dict(topic='testtopic'), prepare={}, expect=dict(topic='testtopic'))), ('prepare_topic', dict(ctor={}, prepare=dict(topic='testtopic'), expect=dict(topic='testtopic'))), ('prepare_topic_none', dict(ctor=dict(topic='testtopic'), prepare=dict(topic=None), expect={})), ('both_topic', dict(ctor=dict(topic='ctortopic'), prepare=dict(topic='testtopic'), expect=dict(topic='testtopic'))), ('ctor_namespace', dict(ctor=dict(namespace='testnamespace'), prepare={}, expect=dict(namespace='testnamespace'))), ('prepare_namespace', dict(ctor={}, prepare=dict(namespace='testnamespace'), expect=dict(namespace='testnamespace'))), ('prepare_namespace_none', dict(ctor=dict(namespace='testnamespace'), prepare=dict(namespace=None), expect={})), ('both_namespace', dict(ctor=dict(namespace='ctornamespace'), prepare=dict(namespace='testnamespace'), expect=dict(namespace='testnamespace'))), ('ctor_version', dict(ctor=dict(version='1.1'), prepare={}, expect=dict(version='1.1'))), ('prepare_version', dict(ctor={}, prepare=dict(version='1.1'), expect=dict(version='1.1'))), ('prepare_version_none', dict(ctor=dict(version='1.1'), prepare=dict(version=None), expect={})), ('both_version', dict(ctor=dict(version='ctorversion'), prepare=dict(version='1.1'), expect=dict(version='1.1'))), ('ctor_server', dict(ctor=dict(server='testserver'), prepare={}, expect=dict(server='testserver'))), ('prepare_server', dict(ctor={}, prepare=dict(server='testserver'), expect=dict(server='testserver'))), ('prepare_server_none', dict(ctor=dict(server='testserver'), prepare=dict(server=None), expect={})), ('both_server', dict(ctor=dict(server='ctorserver'), prepare=dict(server='testserver'), expect=dict(server='testserver'))), ('ctor_fanout', dict(ctor=dict(fanout=True), prepare={}, expect=dict(fanout=True))), ('prepare_fanout', dict(ctor={}, prepare=dict(fanout=True), expect=dict(fanout=True))), ('prepare_fanout_none', dict(ctor=dict(fanout=True), prepare=dict(fanout=None), expect={})), ('both_fanout', dict(ctor=dict(fanout=True), prepare=dict(fanout=False), expect=dict(fanout=False)))]
    _prepare = [('single_prepare', dict(double_prepare=False)), ('double_prepare', dict(double_prepare=True))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._base, cls._prepare)

    def setUp(self):
        super(TestCastToTarget, self).setUp(conf=cfg.ConfigOpts())

    def test_cast_to_target(self):
        target = oslo_messaging.Target(**self.ctor)
        expect_target = oslo_messaging.Target(**self.expect)
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        client = oslo_messaging.get_rpc_client(transport, target)
        transport._send = mock.Mock()
        msg = dict(method='foo', args={})
        if 'namespace' in self.expect:
            msg['namespace'] = self.expect['namespace']
        if 'version' in self.expect:
            msg['version'] = self.expect['version']
        if self.prepare:
            client = client.prepare(**self.prepare)
            if self.double_prepare:
                client = client.prepare(**self.prepare)
        client.cast({}, 'foo')
        transport._send.assert_called_once_with(expect_target, {}, msg, retry=None, transport_options=None)