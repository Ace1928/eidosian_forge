from boto.resultset import ResultSet
from boto.ec2.tag import Tag
from boto.ec2.ec2object import TaggedEC2Object
class VolumeAttribute(object):

    def __init__(self, parent=None):
        self.id = None
        self._key_name = None
        self.attrs = {}

    def startElement(self, name, attrs, connection):
        if name == 'autoEnableIO':
            self._key_name = name
        return None

    def endElement(self, name, value, connection):
        if name == 'value':
            if value.lower() == 'true':
                self.attrs[self._key_name] = True
            else:
                self.attrs[self._key_name] = False
        elif name == 'volumeId':
            self.id = value
        else:
            setattr(self, name, value)