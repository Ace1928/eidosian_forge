from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class PortRange(object):
    """
    Define the port range for the ACL entry if it is tcp / udp
    """

    def __init__(self, connection=None):
        self.from_port = None
        self.to_port = None

    def __repr__(self):
        return 'PortRange:(%s-%s)' % (self.from_port, self.to_port)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'from':
            self.from_port = value
        elif name == 'to':
            self.to_port = value