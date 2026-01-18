from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class Icmp(object):
    """
    Defines the ICMP code and type.
    """

    def __init__(self, connection=None):
        self.code = None
        self.type = None

    def __repr__(self):
        return 'Icmp::code:%s, type:%s)' % (self.code, self.type)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'code':
            self.code = value
        elif name == 'type':
            self.type = value