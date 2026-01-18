from unittest import mock
from openstackclient.compute.v2 import host
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.host_show')
class TestHostShow(compute_fakes.TestComputev2):
    _host = compute_fakes.create_one_host()

    def setUp(self):
        super(TestHostShow, self).setUp()
        output_data = {'resource': {'host': self._host['host'], 'project': self._host['project'], 'cpu': self._host['cpu'], 'memory_mb': self._host['memory_mb'], 'disk_gb': self._host['disk_gb']}}
        self.compute_sdk_client.get.return_value = fakes.FakeResponse(data={'host': [output_data]})
        self.columns = ('Host', 'Project', 'CPU', 'Memory MB', 'Disk GB')
        self.data = [(self._host['host'], self._host['project'], self._host['cpu'], self._host['memory_mb'], self._host['disk_gb'])]
        self.cmd = host.ShowHost(self.app, None)

    def test_host_show_no_option(self, h_mock):
        h_mock.host_show.return_value = [self._host]
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_host_show_with_option(self, h_mock):
        h_mock.return_value = [self._host]
        arglist = [self._host['host_name']]
        verifylist = [('host', self._host['host_name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.get.assert_called_with('/os-hosts/' + self._host['host_name'], microversion='2.1')
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))