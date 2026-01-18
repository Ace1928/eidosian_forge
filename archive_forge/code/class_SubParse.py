import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
class SubParse(dict):

    def __init__(self, section, parent=None):
        dict.__init__(self)
        self.section = section

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name != self.section:
            self[name] = value