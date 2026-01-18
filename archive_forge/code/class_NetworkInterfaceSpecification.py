from boto.exception import BotoClientError
from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.group import Group
class NetworkInterfaceSpecification(object):

    def __init__(self, network_interface_id=None, device_index=None, subnet_id=None, description=None, private_ip_address=None, groups=None, delete_on_termination=None, private_ip_addresses=None, secondary_private_ip_address_count=None, associate_public_ip_address=None):
        self.network_interface_id = network_interface_id
        self.device_index = device_index
        self.subnet_id = subnet_id
        self.description = description
        self.private_ip_address = private_ip_address
        self.groups = groups
        self.delete_on_termination = delete_on_termination
        self.private_ip_addresses = private_ip_addresses
        self.secondary_private_ip_address_count = secondary_private_ip_address_count
        self.associate_public_ip_address = associate_public_ip_address