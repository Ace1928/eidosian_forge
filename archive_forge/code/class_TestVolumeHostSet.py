from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_host
class TestVolumeHostSet(TestVolumeHost):
    service = volume_fakes.create_one_service()

    def setUp(self):
        super().setUp()
        self.host_mock.freeze_host.return_value = None
        self.host_mock.thaw_host.return_value = None
        self.cmd = volume_host.SetVolumeHost(self.app, None)

    def test_volume_host_set_nothing(self):
        arglist = [self.service.host]
        verifylist = [('host', self.service.host)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.host_mock.freeze_host.assert_not_called()
        self.host_mock.thaw_host.assert_not_called()
        self.assertIsNone(result)

    def test_volume_host_set_enable(self):
        arglist = ['--enable', self.service.host]
        verifylist = [('enable', True), ('host', self.service.host)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.host_mock.thaw_host.assert_called_with(self.service.host)
        self.host_mock.freeze_host.assert_not_called()
        self.assertIsNone(result)

    def test_volume_host_set_disable(self):
        arglist = ['--disable', self.service.host]
        verifylist = [('disable', True), ('host', self.service.host)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.host_mock.freeze_host.assert_called_with(self.service.host)
        self.host_mock.thaw_host.assert_not_called()
        self.assertIsNone(result)