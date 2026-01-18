from boto.ec2.elb.listelement import ListElement
from boto.ec2.blockdevicemapping import BlockDeviceMapping as BDM
from boto.resultset import ResultSet
import boto.utils
import base64
class BlockDeviceMapping(object):

    def __init__(self, connection=None, device_name=None, virtual_name=None, ebs=None, no_device=None):
        self.connection = connection
        self.device_name = device_name
        self.virtual_name = virtual_name
        self.ebs = ebs
        self.no_device = no_device

    def __repr__(self):
        return 'BlockDeviceMapping(%s, %s)' % (self.device_name, self.virtual_name)

    def startElement(self, name, attrs, connection):
        if name == 'Ebs':
            self.ebs = Ebs(self)
            return self.ebs

    def endElement(self, name, value, connection):
        if name == 'DeviceName':
            self.device_name = value
        elif name == 'VirtualName':
            self.virtual_name = value
        elif name == 'NoDevice':
            self.no_device = bool(value)