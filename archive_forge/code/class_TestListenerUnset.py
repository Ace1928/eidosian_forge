import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListenerUnset(TestListener):
    PARAMETERS = ('name', 'description', 'connection_limit', 'default_pool_id', 'default_tls_container_ref', 'sni_container_refs', 'insert_headers', 'timeout_client_data', 'timeout_member_connect', 'timeout_member_data', 'timeout_tcp_inspect', 'client_ca_tls_container_ref', 'client_authentication', 'client_crl_container_ref', 'allowed_cidrs', 'tls_versions', 'tls_ciphers')

    def setUp(self):
        super().setUp()
        self.cmd = listener.UnsetListener(self.app, None)

    def test_listener_unset_name(self):
        self._test_listener_unset_param('name')

    def test_listener_unset_name_wait(self):
        self._test_listener_unset_param_wait('name')

    def test_listener_unset_description(self):
        self._test_listener_unset_param('description')

    def test_listener_unset_connection_limit(self):
        self._test_listener_unset_param('connection_limit')

    def test_listener_unset_default_pool(self):
        self._test_listener_unset_param('default_pool')

    def test_listener_unset_default_tls_container_ref(self):
        self._test_listener_unset_param('default_tls_container_ref')

    def test_listener_unset_sni_container_refs(self):
        self._test_listener_unset_param('sni_container_refs')

    def test_listener_unset_insert_headers(self):
        self._test_listener_unset_param('insert_headers')

    def test_listener_unset_timeout_client_data(self):
        self._test_listener_unset_param('timeout_client_data')

    def test_listener_unset_timeout_member_connect(self):
        self._test_listener_unset_param('timeout_member_connect')

    def test_listener_unset_timeout_member_data(self):
        self._test_listener_unset_param('timeout_member_data')

    def test_listener_unset_timeout_tcp_inspect(self):
        self._test_listener_unset_param('timeout_tcp_inspect')

    def test_listener_unset_client_ca_tls_container_ref(self):
        self._test_listener_unset_param('client_ca_tls_container_ref')

    def test_listener_unset_client_authentication(self):
        self._test_listener_unset_param('client_authentication')

    def test_listener_unset_client_crl_container_ref(self):
        self._test_listener_unset_param('client_crl_container_ref')

    def test_listener_unset_allowed_cidrs(self):
        self._test_listener_unset_param('allowed_cidrs')

    def test_listener_unset_tls_versions(self):
        self._test_listener_unset_param('tls_versions')

    def test_listener_unset_tls_ciphers(self):
        self._test_listener_unset_param('tls_ciphers')

    def test_listener_unset_hsts_max_age(self):
        self._test_listener_unset_param('hsts_max_age')

    def test_listener_unset_hsts_include_subdomains(self):
        self._test_listener_unset_param('hsts_include_subdomains')

    def test_listener_unset_hsts_preload(self):
        self._test_listener_unset_param('hsts_preload')

    def _test_listener_unset_param(self, param):
        self.api_mock.listener_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._listener.id, '--%s' % arg_param]
        if param == 'default_pool':
            param = 'default_pool_id'
        ref_body = {'listener': {param: None}}
        verifylist = [('listener', self._listener.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_listener_unset_param_wait(self, param, mock_wait):
        self.api_mock.listener_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._listener.id, '--%s' % arg_param, '--wait']
        if param == 'default_pool':
            param = 'default_pool_id'
        ref_body = {'listener': {param: None}}
        verifylist = [('listener', self._listener.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._listener.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_listener_unset_all(self):
        self.api_mock.listener_set.reset_mock()
        ref_body = {'listener': {x: None for x in self.PARAMETERS}}
        arglist = [self._listener.id]
        for ref_param in self.PARAMETERS:
            if ref_param == 'default_pool_id':
                ref_param = 'default_pool'
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('listener', self._listener.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json=ref_body)

    def test_listener_unset_none(self):
        self.api_mock.listener_set.reset_mock()
        arglist = [self._listener.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('listener', self._listener.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_not_called()

    def test_listener_unset_tag(self):
        self.api_mock.listener_set.reset_mock()
        self.api_mock.listener_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._listener.id, '--tag', 'foo']
        verifylist = [('listener', self._listener.id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json={'listener': {'tags': ['bar']}})

    def test_listener_unset_all_tag(self):
        self.api_mock.listener_set.reset_mock()
        self.api_mock.listener_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._listener.id, '--all-tag']
        verifylist = [('listener', self._listener.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_set.assert_called_once_with(self._listener.id, json={'listener': {'tags': []}})