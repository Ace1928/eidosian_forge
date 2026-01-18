from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.ec2.blockdevicemapping import BlockDeviceMapping
class BillingProducts(list):

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'billingProduct':
            self.append(value)