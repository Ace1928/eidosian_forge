from tests.compat import mock, unittest
from boto.exception import BotoClientError
from boto.ec2.networkinterface import NetworkInterfaceCollection
from boto.ec2.networkinterface import NetworkInterfaceSpecification
from boto.ec2.networkinterface import PrivateIPAddress
from boto.ec2.networkinterface import Attachment, NetworkInterface
class TestNetworkInterfaceCollection(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.private_ip_address1 = PrivateIPAddress(private_ip_address='10.0.0.10', primary=False)
        self.private_ip_address2 = PrivateIPAddress(private_ip_address='10.0.0.11', primary=False)
        self.network_interfaces_spec1 = NetworkInterfaceSpecification(device_index=1, subnet_id='subnet_id', description='description1', private_ip_address='10.0.0.54', delete_on_termination=False, private_ip_addresses=[self.private_ip_address1, self.private_ip_address2])
        self.private_ip_address3 = PrivateIPAddress(private_ip_address='10.0.1.10', primary=False)
        self.private_ip_address4 = PrivateIPAddress(private_ip_address='10.0.1.11', primary=False)
        self.network_interfaces_spec2 = NetworkInterfaceSpecification(device_index=2, subnet_id='subnet_id2', description='description2', groups=['group_id1', 'group_id2'], private_ip_address='10.0.1.54', delete_on_termination=False, private_ip_addresses=[self.private_ip_address3, self.private_ip_address4])
        self.network_interfaces_spec3 = NetworkInterfaceSpecification(device_index=0, subnet_id='subnet_id2', description='description2', groups=['group_id1', 'group_id2'], private_ip_address='10.0.1.54', delete_on_termination=False, private_ip_addresses=[self.private_ip_address3, self.private_ip_address4], associate_public_ip_address=True)

    def test_param_serialization(self):
        collection = NetworkInterfaceCollection(self.network_interfaces_spec1, self.network_interfaces_spec2)
        params = {}
        collection.build_list_params(params)
        self.assertDictEqual(params, {'NetworkInterface.0.DeviceIndex': '1', 'NetworkInterface.0.DeleteOnTermination': 'false', 'NetworkInterface.0.Description': 'description1', 'NetworkInterface.0.PrivateIpAddress': '10.0.0.54', 'NetworkInterface.0.SubnetId': 'subnet_id', 'NetworkInterface.0.PrivateIpAddresses.0.Primary': 'false', 'NetworkInterface.0.PrivateIpAddresses.0.PrivateIpAddress': '10.0.0.10', 'NetworkInterface.0.PrivateIpAddresses.1.Primary': 'false', 'NetworkInterface.0.PrivateIpAddresses.1.PrivateIpAddress': '10.0.0.11', 'NetworkInterface.1.DeviceIndex': '2', 'NetworkInterface.1.Description': 'description2', 'NetworkInterface.1.DeleteOnTermination': 'false', 'NetworkInterface.1.PrivateIpAddress': '10.0.1.54', 'NetworkInterface.1.SubnetId': 'subnet_id2', 'NetworkInterface.1.SecurityGroupId.0': 'group_id1', 'NetworkInterface.1.SecurityGroupId.1': 'group_id2', 'NetworkInterface.1.PrivateIpAddresses.0.Primary': 'false', 'NetworkInterface.1.PrivateIpAddresses.0.PrivateIpAddress': '10.0.1.10', 'NetworkInterface.1.PrivateIpAddresses.1.Primary': 'false', 'NetworkInterface.1.PrivateIpAddresses.1.PrivateIpAddress': '10.0.1.11'})

    def test_add_prefix_to_serialization(self):
        collection = NetworkInterfaceCollection(self.network_interfaces_spec1, self.network_interfaces_spec2)
        params = {}
        collection.build_list_params(params, prefix='LaunchSpecification.')
        self.assertDictEqual(params, {'LaunchSpecification.NetworkInterface.0.DeviceIndex': '1', 'LaunchSpecification.NetworkInterface.0.DeleteOnTermination': 'false', 'LaunchSpecification.NetworkInterface.0.Description': 'description1', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddress': '10.0.0.54', 'LaunchSpecification.NetworkInterface.0.SubnetId': 'subnet_id', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.0.Primary': 'false', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.0.PrivateIpAddress': '10.0.0.10', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.1.Primary': 'false', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.1.PrivateIpAddress': '10.0.0.11', 'LaunchSpecification.NetworkInterface.1.DeviceIndex': '2', 'LaunchSpecification.NetworkInterface.1.Description': 'description2', 'LaunchSpecification.NetworkInterface.1.DeleteOnTermination': 'false', 'LaunchSpecification.NetworkInterface.1.PrivateIpAddress': '10.0.1.54', 'LaunchSpecification.NetworkInterface.1.SubnetId': 'subnet_id2', 'LaunchSpecification.NetworkInterface.1.SecurityGroupId.0': 'group_id1', 'LaunchSpecification.NetworkInterface.1.SecurityGroupId.1': 'group_id2', 'LaunchSpecification.NetworkInterface.1.PrivateIpAddresses.0.Primary': 'false', 'LaunchSpecification.NetworkInterface.1.PrivateIpAddresses.0.PrivateIpAddress': '10.0.1.10', 'LaunchSpecification.NetworkInterface.1.PrivateIpAddresses.1.Primary': 'false', 'LaunchSpecification.NetworkInterface.1.PrivateIpAddresses.1.PrivateIpAddress': '10.0.1.11'})

    def test_cant_use_public_ip(self):
        collection = NetworkInterfaceCollection(self.network_interfaces_spec3, self.network_interfaces_spec1)
        params = {}
        with self.assertRaises(BotoClientError):
            collection.build_list_params(params, prefix='LaunchSpecification.')
        self.network_interfaces_spec3.device_index = 1
        collection = NetworkInterfaceCollection(self.network_interfaces_spec3)
        params = {}
        with self.assertRaises(BotoClientError):
            collection.build_list_params(params, prefix='LaunchSpecification.')

    def test_public_ip(self):
        collection = NetworkInterfaceCollection(self.network_interfaces_spec3)
        params = {}
        collection.build_list_params(params, prefix='LaunchSpecification.')
        self.assertDictEqual(params, {'LaunchSpecification.NetworkInterface.0.AssociatePublicIpAddress': 'true', 'LaunchSpecification.NetworkInterface.0.DeviceIndex': '0', 'LaunchSpecification.NetworkInterface.0.DeleteOnTermination': 'false', 'LaunchSpecification.NetworkInterface.0.Description': 'description2', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddress': '10.0.1.54', 'LaunchSpecification.NetworkInterface.0.SubnetId': 'subnet_id2', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.0.Primary': 'false', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.0.PrivateIpAddress': '10.0.1.10', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.1.Primary': 'false', 'LaunchSpecification.NetworkInterface.0.PrivateIpAddresses.1.PrivateIpAddress': '10.0.1.11', 'LaunchSpecification.NetworkInterface.0.SecurityGroupId.0': 'group_id1', 'LaunchSpecification.NetworkInterface.0.SecurityGroupId.1': 'group_id2'})