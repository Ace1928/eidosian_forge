import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestHealthMonitorUnset(TestHealthMonitor):
    PARAMETERS = ('name', 'domain_name', 'expected_codes', 'http_method', 'http_version', 'max_retries_down', 'url_path')

    def setUp(self):
        super().setUp()
        self.cmd = health_monitor.UnsetHealthMonitor(self.app, None)

    def test_hm_unset_domain_name(self):
        self._test_hm_unset_param('domain_name')

    def test_hm_unset_expected_codes(self):
        self._test_hm_unset_param('expected_codes')

    def test_hm_unset_http_method(self):
        self._test_hm_unset_param('http_method')

    def test_hm_unset_http_version(self):
        self._test_hm_unset_param('http_version')

    def test_hm_unset_max_retries_down(self):
        self._test_hm_unset_param('max_retries_down')

    def test_hm_unset_name(self):
        self._test_hm_unset_param('name')

    def test_hm_unset_name_wait(self):
        self._test_hm_unset_param_wait('name')

    def test_hm_unset_url_path(self):
        self._test_hm_unset_param('url_path')

    def _test_hm_unset_param(self, param):
        self.api_mock.health_monitor_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._hm.id, '--%s' % arg_param]
        ref_body = {'healthmonitor': {param: None}}
        verifylist = [('health_monitor', self._hm.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_hm_unset_param_wait(self, param, mock_wait):
        self.api_mock.health_monitor_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._hm.id, '--%s' % arg_param, '--wait']
        ref_body = {'healthmonitor': {param: None}}
        verifylist = [('health_monitor', self._hm.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._hm.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_hm_unset_all(self):
        self.api_mock.health_monitor_set.reset_mock()
        ref_body = {'healthmonitor': {x: None for x in self.PARAMETERS}}
        arglist = [self._hm.id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('health_monitor', self._hm.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json=ref_body)

    def test_hm_unset_none(self):
        self.api_mock.health_monitor_set.reset_mock()
        arglist = [self._hm.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('health_monitor', self._hm.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_not_called()

    def test_health_monitor_unset_tag(self):
        self.api_mock.health_monitor_set.reset_mock()
        self.api_mock.health_monitor_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._hm.id, '--tag', 'foo']
        verifylist = [('health_monitor', self._hm.id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json={'healthmonitor': {'tags': ['bar']}})

    def test_health_monitor_unset_all_tag(self):
        self.api_mock.health_monitor_set.reset_mock()
        self.api_mock.health_monitor_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._hm.id, '--all-tag']
        verifylist = [('health_monitor', self._hm.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_set.assert_called_once_with(self._hm.id, json={'healthmonitor': {'tags': []}})