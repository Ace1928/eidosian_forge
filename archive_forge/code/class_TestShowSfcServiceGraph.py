from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestShowSfcServiceGraph(fakes.TestNeutronClientOSCV2):
    _sg = fakes.FakeSfcServiceGraph.create_sfc_service_graph()
    columns = ('ID', 'Name', 'Branching Points')
    columns_long = ('Branching Points', 'Description', 'ID', 'Name', 'Project')
    data = (_sg['id'], _sg['name'], _sg['port_chains'])
    data_long = (_sg['port_chains'], _sg['description'], _sg['id'], _sg['name'], _sg['project_id'])
    _service_graph = _sg
    _service_graph_id = _sg['id']

    def setUp(self):
        super(TestShowSfcServiceGraph, self).setUp()
        self.network.get_sfc_service_graph = mock.Mock(return_value=self._service_graph)
        self.cmd = sfc_service_graph.ShowSfcServiceGraph(self.app, self.namespace)

    def test_service_graph_show(self):
        client = self.app.client_manager.network
        mock_service_graph_show = client.get_sfc_service_graph
        arglist = [self._service_graph_id]
        verifylist = [('service_graph', self._service_graph_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_service_graph_show.assert_called_once_with(self._service_graph_id)
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data_long, data)