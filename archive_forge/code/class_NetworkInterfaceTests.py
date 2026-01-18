from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
class NetworkInterfaceTests(unittest.TestCase):

    def setUp(self):
        self.attachment = Attachment()
        self.attachment.id = 'eni-attach-1'
        self.attachment.instance_id = 10
        self.attachment.status = 'some status'
        self.attachment.device_index = 100
        self.eni_one = NetworkInterface()
        self.eni_one.id = 'eni-1'
        self.eni_one.status = 'one_status'
        self.eni_one.attachment = self.attachment
        self.eni_two = NetworkInterface()
        self.eni_two.connection = mock.Mock()
        self.eni_two.id = 'eni-2'
        self.eni_two.status = 'two_status'
        self.eni_two.attachment = None

    def test_update_with_validate_true_raises_value_error(self):
        self.eni_one.connection = mock.Mock()
        self.eni_one.connection.get_all_network_interfaces.return_value = []
        with self.assertRaisesRegexp(ValueError, '^eni-1 is not a valid ENI ID$'):
            self.eni_one.update(True)

    def test_update_with_result_set_greater_than_0_updates_dict(self):
        self.eni_two.connection.get_all_network_interfaces.return_value = [self.eni_one]
        self.eni_two.update()
        assert all([self.eni_two.status == 'one_status', self.eni_two.id == 'eni-1', self.eni_two.attachment == self.attachment])

    def test_update_returns_status(self):
        self.eni_one.connection = mock.Mock()
        self.eni_one.connection.get_all_network_interfaces.return_value = [self.eni_two]
        retval = self.eni_one.update()
        self.assertEqual(retval, 'two_status')

    def test_attach_calls_attach_eni(self):
        self.eni_one.connection = mock.Mock()
        self.eni_one.attach('instance_id', 11)
        self.eni_one.connection.attach_network_interface.assert_called_with('eni-1', 'instance_id', 11, dry_run=False)

    def test_detach_calls_detach_network_interface(self):
        self.eni_one.connection = mock.Mock()
        self.eni_one.detach()
        self.eni_one.connection.detach_network_interface.assert_called_with('eni-attach-1', False, dry_run=False)

    def test_detach_with_no_attach_data(self):
        self.eni_two.connection = mock.Mock()
        self.eni_two.detach()
        self.eni_two.connection.detach_network_interface.assert_called_with(None, False, dry_run=False)

    def test_detach_with_force_calls_detach_network_interface_with_force(self):
        self.eni_one.connection = mock.Mock()
        self.eni_one.detach(True)
        self.eni_one.connection.detach_network_interface.assert_called_with('eni-attach-1', True, dry_run=False)