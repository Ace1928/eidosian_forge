import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListenerCreate(TestListener):

    def setUp(self):
        super().setUp()
        self.api_mock.listener_create.return_value = {'listener': self.listener_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = listener.CreateListener(self.app, None)

    @mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
    def test_listener_create(self, mock_client):
        mock_client.return_value = self.listener_info
        arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'HTTP', '--protocol-port', '80']
        verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'HTTP'), ('protocol_port', 80)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertIsNone(parsed_args.hsts_include_subdomains)
        self.assertIsNone(parsed_args.hsts_preload)
        self.assertIsNone(parsed_args.hsts_max_age)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})

    @mock.patch('osc_lib.utils.wait_for_status')
    @mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
    def test_listener_create_wait(self, mock_client, mock_wait):
        self.listener_info['loadbalancers'] = [{'id': 'mock_lb_id'}]
        mock_client.return_value = self.listener_info
        self.api_mock.listener_show.return_value = self.listener_info
        arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'HTTP', '--protocol-port', '80', '--wait']
        verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'HTTP'), ('protocol_port', 80), ('wait', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self.listener_info['loadbalancers'][0]['id'], sleep_time=mock.ANY, status_field='provisioning_status')

    @mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
    def test_tls_listener_create(self, mock_client):
        mock_client.return_value = self.listener_info
        arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'TERMINATED_HTTPS'.lower(), '--protocol-port', '443', '--sni-container-refs', self._listener.sni_container_refs[0], self._listener.sni_container_refs[1], '--default-tls-container-ref', self._listener.default_tls_container_ref, '--client-ca-tls-container-ref', self._listener.client_ca_tls_container_ref, '--client-authentication', self._listener.client_authentication, '--client-crl-container-ref', self._listener.client_crl_container_ref, '--tls-ciphers', self._listener.tls_ciphers, '--tls-version', self._listener.tls_versions[0], '--tls-version', self._listener.tls_versions[1], '--alpn-protocol', self._listener.alpn_protocols[0], '--alpn-protocol', self._listener.alpn_protocols[1], '--hsts-max-age', '12000000', '--hsts-include-subdomains', '--hsts-preload']
        verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'TERMINATED_HTTPS'), ('protocol_port', 443), ('sni_container_refs', self._listener.sni_container_refs), ('default_tls_container_ref', self._listener.default_tls_container_ref), ('client_ca_tls_container_ref', self._listener.client_ca_tls_container_ref), ('client_authentication', self._listener.client_authentication), ('client_crl_container_ref', self._listener.client_crl_container_ref), ('tls_ciphers', self._listener.tls_ciphers), ('tls_versions', self._listener.tls_versions), ('alpn_protocols', self._listener.alpn_protocols), ('hsts_max_age', 12000000), ('hsts_include_subdomains', True), ('hsts_preload', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
    def test_listener_create_timeouts(self, mock_client):
        mock_client.return_value = self.listener_info
        arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'HTTP', '--protocol-port', '80', '--timeout-client-data', '123', '--timeout-member-connect', '234', '--timeout-member-data', '345', '--timeout-tcp-inspect', '456']
        verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'HTTP'), ('protocol_port', 80), ('timeout_client_data', 123), ('timeout_member_connect', 234), ('timeout_member_data', 345), ('timeout_tcp_inspect', 456)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})

    @mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
    def test_listener_create_with_tag(self, mock_client):
        mock_client.return_value = self.listener_info
        arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'HTTP', '--protocol-port', '80', '--tag', 'foo']
        verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'HTTP'), ('protocol_port', 80), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})