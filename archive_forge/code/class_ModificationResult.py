from boto.resultset import ResultSet
from boto.ec2.ec2object import EC2Object
from boto.utils import parse_ts
class ModificationResult(object):

    def __init__(self, connection=None, modification_id=None, availability_zone=None, platform=None, instance_count=None, instance_type=None):
        self.connection = connection
        self.modification_id = modification_id
        self.availability_zone = availability_zone
        self.platform = platform
        self.instance_count = instance_count
        self.instance_type = instance_type

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'reservedInstancesModificationId':
            self.modification_id = value
        elif name == 'availabilityZone':
            self.availability_zone = value
        elif name == 'platform':
            self.platform = value
        elif name == 'instanceCount':
            self.instance_count = int(value)
        elif name == 'instanceType':
            self.instance_type = value
        else:
            setattr(self, name, value)