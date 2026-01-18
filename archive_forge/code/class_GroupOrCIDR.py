from boto.ec2.ec2object import TaggedEC2Object
from boto.exception import BotoClientError
class GroupOrCIDR(object):

    def __init__(self, parent=None):
        self.owner_id = None
        self.group_id = None
        self.name = None
        self.cidr_ip = None

    def __repr__(self):
        if self.cidr_ip:
            return '%s' % self.cidr_ip
        else:
            return '%s-%s' % (self.name or self.group_id, self.owner_id)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'userId':
            self.owner_id = value
        elif name == 'groupId':
            self.group_id = value
        elif name == 'groupName':
            self.name = value
        if name == 'cidrIp':
            self.cidr_ip = value
        else:
            setattr(self, name, value)