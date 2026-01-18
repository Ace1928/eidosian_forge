import boto
from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.resultset import ResultSet
from boto.ec2.address import Address
from boto.ec2.blockdevicemapping import BlockDeviceMapping
from boto.ec2.image import ProductCodes
from boto.ec2.networkinterface import NetworkInterface
from boto.ec2.group import Group
import base64
class InstanceAttribute(dict):
    ValidValues = ['instanceType', 'kernel', 'ramdisk', 'userData', 'disableApiTermination', 'instanceInitiatedShutdownBehavior', 'rootDeviceName', 'blockDeviceMapping', 'sourceDestCheck', 'groupSet']

    def __init__(self, parent=None):
        dict.__init__(self)
        self.instance_id = None
        self.request_id = None
        self._current_value = None

    def startElement(self, name, attrs, connection):
        if name == 'blockDeviceMapping':
            self[name] = BlockDeviceMapping()
            return self[name]
        elif name == 'groupSet':
            self[name] = ResultSet([('item', Group)])
            return self[name]
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'instanceId':
            self.instance_id = value
        elif name == 'requestId':
            self.request_id = value
        elif name == 'value':
            if value == 'true':
                value = True
            elif value == 'false':
                value = False
            self._current_value = value
        elif name in self.ValidValues:
            self[name] = self._current_value