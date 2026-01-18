from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_port_chain
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestDeleteSfcPortChain(fakes.TestNeutronClientOSCV2):
    _port_chain = fakes.FakeSfcPortChain.create_port_chains(count=1)

    def setUp(self):
        super(TestDeleteSfcPortChain, self).setUp()
        self.network.delete_sfc_port_chain = mock.Mock(return_value=None)
        self.cmd = sfc_port_chain.DeleteSfcPortChain(self.app, self.namespace)

    def test_delete_port_chain(self):
        client = self.app.client_manager.network
        mock_port_chain_delete = client.delete_sfc_port_chain
        arglist = [self._port_chain[0]['id']]
        verifylist = [('port_chain', [self._port_chain[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        mock_port_chain_delete.assert_called_once_with(self._port_chain[0]['id'])
        self.assertIsNone(result)

    def test_delete_multiple_port_chains_with_exception(self):
        client = self.app.client_manager.network
        target1 = self._port_chain[0]['id']
        arglist = [target1]
        verifylist = [('port_chain', [target1])]
        client.find_sfc_port_chain.side_effect = [target1, exceptions.CommandError]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = '1 of 2 port chain(s) failed to delete.'
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual(msg, str(e))