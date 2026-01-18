from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestCanSendVersion(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(cap=None, prepare_cap=_notset, version=None, prepare_version=_notset, can_send_version=_notset, can_send=True)), ('ctor_cap_ok', dict(cap='1.1', prepare_cap=_notset, version='1.0', prepare_version=_notset, can_send_version=_notset, can_send=True)), ('ctor_cap_override_ok', dict(cap='2.0', prepare_cap='1.1', version='1.0', prepare_version='1.0', can_send_version=_notset, can_send=True)), ('ctor_cap_override_none_ok', dict(cap='1.1', prepare_cap=None, version='1.0', prepare_version=_notset, can_send_version=_notset, can_send=True)), ('ctor_cap_can_send_ok', dict(cap='1.1', prepare_cap=None, version='1.0', prepare_version=_notset, can_send_version='1.1', can_send=True)), ('ctor_cap_can_send_none_ok', dict(cap='1.1', prepare_cap=None, version='1.0', prepare_version=_notset, can_send_version=None, can_send=True)), ('ctor_cap_minor_fail', dict(cap='1.0', prepare_cap=_notset, version='1.1', prepare_version=_notset, can_send_version=_notset, can_send=False)), ('ctor_cap_major_fail', dict(cap='2.0', prepare_cap=_notset, version=None, prepare_version='1.0', can_send_version=_notset, can_send=False)), ('ctor_cap_none_version_ok', dict(cap=None, prepare_cap=_notset, version='1.0', prepare_version=_notset, can_send_version=_notset, can_send=True)), ('ctor_cap_version_none_fail', dict(cap='1.0', prepare_cap=_notset, version=None, prepare_version=_notset, can_send_version=_notset, can_send=False)), ('ctor_cap_version_can_send_none_fail', dict(cap='1.0', prepare_cap=_notset, version='1.0', prepare_version=_notset, can_send_version=None, can_send=False))]

    def test_version_cap(self):
        self.config(rpc_response_timeout=None)
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(version=self.version)
        client = oslo_messaging.get_rpc_client(transport, target, version_cap=self.cap)
        prep_kwargs = {}
        if self.prepare_cap is not _notset:
            prep_kwargs['version_cap'] = self.prepare_cap
        if self.prepare_version is not _notset:
            prep_kwargs['version'] = self.prepare_version
        if prep_kwargs:
            client = client.prepare(**prep_kwargs)
        if self.can_send_version is not _notset:
            can_send = client.can_send_version(version=self.can_send_version)
            call_context_can_send = client.prepare().can_send_version(version=self.can_send_version)
            self.assertEqual(can_send, call_context_can_send)
        else:
            can_send = client.can_send_version()
        self.assertEqual(self.can_send, can_send)

    def test_invalid_version_type(self):
        target = oslo_messaging.Target(topic='sometopic')
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        client = oslo_messaging.get_rpc_client(transport, target)
        self.assertRaises(exceptions.MessagingException, client.prepare, version='5')
        self.assertRaises(exceptions.MessagingException, client.prepare, version='5.a')
        self.assertRaises(exceptions.MessagingException, client.prepare, version='5.5.a')