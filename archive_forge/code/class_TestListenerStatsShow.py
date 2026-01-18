import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestListenerStatsShow(TestListener):

    def setUp(self):
        super().setUp()
        listener_stats_info = {'stats': {'bytes_in': '0'}}
        self.api_mock.listener_stats_show.return_value = {'stats': listener_stats_info['stats']}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = listener.ShowListenerStats(self.app, None)

    def test_listener_stats_show(self):
        arglist = [self._listener.id]
        verifylist = [('listener', self._listener.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.listener_stats_show.assert_called_with(listener_id=self._listener.id)