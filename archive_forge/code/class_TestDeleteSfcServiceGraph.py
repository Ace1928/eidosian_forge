from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestDeleteSfcServiceGraph(fakes.TestNeutronClientOSCV2):
    _service_graph = fakes.FakeSfcServiceGraph.create_sfc_service_graphs(count=1)

    def setUp(self):
        super(TestDeleteSfcServiceGraph, self).setUp()
        self.network.delete_sfc_service_graph = mock.Mock(return_value=None)
        self.cmd = sfc_service_graph.DeleteSfcServiceGraph(self.app, self.namespace)

    def test_delete_sfc_service_graph(self):
        client = self.app.client_manager.network
        mock_service_graph_delete = client.delete_sfc_service_graph
        arglist = [self._service_graph[0]['id']]
        verifylist = [('service_graph', [self._service_graph[0]['id']])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        mock_service_graph_delete.assert_called_once_with(self._service_graph[0]['id'])
        self.assertIsNone(result)

    def test_delete_multiple_service_graphs_with_exception(self):
        client = self.app.client_manager.network
        target = self._service_graph[0]['id']
        arglist = [target]
        verifylist = [('service_graph', [target])]
        client.find_sfc_service_graph.side_effect = [target, exceptions.CommandError]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        msg = '1 of 2 service graph(s) failed to delete.'
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual(msg, str(e))