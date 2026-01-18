from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestVersionCap(test_utils.BaseTestCase):
    _call_vs_cast = [('call', dict(call=True)), ('cast', dict(call=False))]
    _cap_scenarios = [('all_none', dict(cap=None, prepare_cap=_notset, version=None, prepare_version=_notset, success=True)), ('ctor_cap_ok', dict(cap='1.1', prepare_cap=_notset, version='1.0', prepare_version=_notset, success=True)), ('ctor_cap_override_ok', dict(cap='2.0', prepare_cap='1.1', version='1.0', prepare_version='1.0', success=True)), ('ctor_cap_override_none_ok', dict(cap='1.1', prepare_cap=None, version='1.0', prepare_version=_notset, success=True)), ('ctor_cap_minor_fail', dict(cap='1.0', prepare_cap=_notset, version='1.1', prepare_version=_notset, success=False)), ('ctor_cap_major_fail', dict(cap='2.0', prepare_cap=_notset, version=None, prepare_version='1.0', success=False)), ('ctor_cap_none_version_ok', dict(cap=None, prepare_cap=_notset, version='1.0', prepare_version=_notset, success=True)), ('ctor_cap_version_none_fail', dict(cap='1.0', prepare_cap=_notset, version=None, prepare_version=_notset, success=False))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._call_vs_cast, cls._cap_scenarios)

    def test_version_cap(self):
        self.config(rpc_response_timeout=None)
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        target = oslo_messaging.Target(version=self.version)
        client = oslo_messaging.get_rpc_client(transport, target, version_cap=self.cap)
        if self.success:
            transport._send = mock.Mock()
            if self.prepare_version is not _notset:
                target = target(version=self.prepare_version)
            msg = dict(method='foo', args={})
            if target.version is not None:
                msg['version'] = target.version
            kwargs = {'retry': None}
            if self.call:
                kwargs['wait_for_reply'] = True
                kwargs['timeout'] = None
                kwargs['call_monitor_timeout'] = None
        prep_kwargs = {}
        if self.prepare_cap is not _notset:
            prep_kwargs['version_cap'] = self.prepare_cap
        if self.prepare_version is not _notset:
            prep_kwargs['version'] = self.prepare_version
        if prep_kwargs:
            client = client.prepare(**prep_kwargs)
        method = client.call if self.call else client.cast
        try:
            method({}, 'foo')
        except Exception as ex:
            self.assertIsInstance(ex, oslo_messaging.RPCVersionCapError, ex)
            self.assertFalse(self.success)
        else:
            self.assertTrue(self.success)
            transport._send.assert_called_once_with(target, {}, msg, transport_options=None, **kwargs)