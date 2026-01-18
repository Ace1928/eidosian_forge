from boto.ec2.ec2object import EC2Object, TaggedEC2Object
from boto.ec2.blockdevicemapping import BlockDeviceMapping
class CopyImage(object):

    def __init__(self, parent=None):
        self._parent = parent
        self.image_id = None

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'imageId':
            self.image_id = value