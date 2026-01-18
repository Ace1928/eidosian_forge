from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestCallTimeout(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(confval=None, ctor=None, prepare=_notset, expect=None, cm=None)), ('confval', dict(confval=21, ctor=None, prepare=_notset, expect=21, cm=None)), ('ctor', dict(confval=None, ctor=21.1, prepare=_notset, expect=21.1, cm=None)), ('ctor_zero', dict(confval=None, ctor=0, prepare=_notset, expect=0, cm=None)), ('prepare', dict(confval=None, ctor=None, prepare=21.1, expect=21.1, cm=None)), ('prepare_override', dict(confval=None, ctor=10.1, prepare=21.1, expect=21.1, cm=None)), ('prepare_zero', dict(confval=None, ctor=None, prepare=0, expect=0, cm=None)), ('call_monitor', dict(confval=None, ctor=None, prepare=60, expect=60, cm=30))]

    def test_call_timeout(self):
        self.config(rpc_response_timeout=self.confval)
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        client = oslo_messaging.get_rpc_client(transport, oslo_messaging.Target(), timeout=self.ctor, call_monitor_timeout=self.cm)
        transport._send = mock.Mock()
        msg = dict(method='foo', args={})
        kwargs = dict(wait_for_reply=True, timeout=self.expect, retry=None, call_monitor_timeout=self.cm, transport_options=None)
        if self.prepare is not _notset:
            client = client.prepare(timeout=self.prepare)
        client.call({}, 'foo')
        transport._send.assert_called_once_with(oslo_messaging.Target(), {}, msg, **kwargs)