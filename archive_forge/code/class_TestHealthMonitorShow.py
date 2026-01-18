import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import health_monitor
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
class TestHealthMonitorShow(TestHealthMonitor):

    def setUp(self):
        super().setUp()
        self.api_mock.health_monitor_show.return_value = {'healthmonitor': self.hm_info}
        lb_client = self.app.client_manager
        lb_client.load_balancer = self.api_mock
        self.cmd = health_monitor.ShowHealthMonitor(self.app, None)

    def test_health_monitor_show(self):
        arglist = [self._hm.id]
        verifylist = [('health_monitor', self._hm.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.api_mock.health_monitor_show.assert_called_with(health_monitor_id=self._hm.id)