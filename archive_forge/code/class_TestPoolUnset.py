import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import pool as pool
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestPoolUnset(TestPool):
    PARAMETERS = ('name', 'description', 'ca_tls_container_ref', 'crl_container_ref', 'session_persistence', 'tls_container_ref', 'tls_versions', 'tls_ciphers')

    def setUp(self):
        super().setUp()
        self.cmd = pool.UnsetPool(self.app, None)

    def test_pool_unset_name(self):
        self._test_pool_unset_param('name')

    def test_pool_unset_name_wait(self):
        self._test_pool_unset_param_wait('name')

    def test_pool_unset_description(self):
        self._test_pool_unset_param('description')

    def test_pool_unset_ca_tls_container_ref(self):
        self._test_pool_unset_param('ca_tls_container_ref')

    def test_pool_unset_crl_container_ref(self):
        self._test_pool_unset_param('crl_container_ref')

    def test_pool_unset_session_persistence(self):
        self._test_pool_unset_param('session_persistence')

    def test_pool_unset_tls_container_ref(self):
        self._test_pool_unset_param('tls_container_ref')

    def test_pool_unset_tls_versions(self):
        self._test_pool_unset_param('tls_versions')

    def test_pool_unset_tls_ciphers(self):
        self._test_pool_unset_param('tls_ciphers')

    def _test_pool_unset_param(self, param):
        self.api_mock.pool_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._po.id, '--%s' % arg_param]
        ref_body = {'pool': {param: None}}
        verifylist = [('pool', self._po.id)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_called_once_with(self._po.id, json=ref_body)

    @mock.patch('osc_lib.utils.wait_for_status')
    def _test_pool_unset_param_wait(self, param, mock_wait):
        self.api_mock.pool_set.reset_mock()
        arg_param = param.replace('_', '-') if '_' in param else param
        arglist = [self._po.id, '--%s' % arg_param, '--wait']
        ref_body = {'pool': {param: None}}
        verifylist = [('pool', self._po.id), ('wait', True)]
        for ref_param in self.PARAMETERS:
            verifylist.append((ref_param, param == ref_param))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_called_once_with(self._po.id, json=ref_body)
        mock_wait.assert_called_once_with(status_f=mock.ANY, res_id=self._po.id, sleep_time=mock.ANY, status_field='provisioning_status')

    def test_pool_unset_all(self):
        self.api_mock.pool_set.reset_mock()
        ref_body = {'pool': {x: None for x in self.PARAMETERS}}
        arglist = [self._po.id]
        for ref_param in self.PARAMETERS:
            arg_param = ref_param.replace('_', '-') if '_' in ref_param else ref_param
            arglist.append('--%s' % arg_param)
        verifylist = list(zip(self.PARAMETERS, [True] * len(self.PARAMETERS)))
        verifylist = [('pool', self._po.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_called_once_with(self._po.id, json=ref_body)

    def test_pool_unset_none(self):
        self.api_mock.pool_set.reset_mock()
        arglist = [self._po.id]
        verifylist = list(zip(self.PARAMETERS, [False] * len(self.PARAMETERS)))
        verifylist = [('pool', self._po.id)] + verifylist
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_not_called()

    def test_pool_unset_tag(self):
        self.api_mock.pool_set.reset_mock()
        self.api_mock.pool_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._po.id, '--tag', 'foo']
        verifylist = [('pool', self._po.id), ('tags', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_called_once_with(self._po.id, json={'pool': {'tags': ['bar']}})

    def test_pool_unset_all_tag(self):
        self.api_mock.pool_set.reset_mock()
        self.api_mock.pool_show.return_value = {'tags': ['foo', 'bar']}
        arglist = [self._po.id, '--all-tag']
        verifylist = [('pool', self._po.id), ('all_tag', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.pool_set.assert_called_once_with(self._po.id, json={'pool': {'tags': []}})