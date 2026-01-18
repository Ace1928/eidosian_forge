import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
class InstancePlacement(object):
    """
    The location where the instance launched.

    :ivar zone: The Availability Zone of the instance.
    :ivar group_name: The name of the placement group the instance is
        in (for cluster compute instances).
    :ivar tenancy: The tenancy of the instance (if the instance is
        running within a VPC). An instance with a tenancy of dedicated
        runs on single-tenant hardware.
    """

    def __init__(self, zone=None, group_name=None, tenancy=None):
        self.zone = zone
        self.group_name = group_name
        self.tenancy = tenancy

    def __repr__(self):
        return self.zone

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'availabilityZone':
            self.zone = value
        elif name == 'groupName':
            self.group_name = value
        elif name == 'tenancy':
            self.tenancy = value
        else:
            setattr(self, name, value)