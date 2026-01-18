from boto.ec2.ec2object import TaggedEC2Object
from boto.exception import BotoClientError
class IPPermissions(object):

    def __init__(self, parent=None):
        self.parent = parent
        self.ip_protocol = None
        self.from_port = None
        self.to_port = None
        self.grants = []

    def __repr__(self):
        return 'IPPermissions:%s(%s-%s)' % (self.ip_protocol, self.from_port, self.to_port)

    def startElement(self, name, attrs, connection):
        if name == 'item':
            self.grants.append(GroupOrCIDR(self))
            return self.grants[-1]
        return None

    def endElement(self, name, value, connection):
        if name == 'ipProtocol':
            self.ip_protocol = value
        elif name == 'fromPort':
            self.from_port = value
        elif name == 'toPort':
            self.to_port = value
        else:
            setattr(self, name, value)

    def add_grant(self, name=None, owner_id=None, cidr_ip=None, group_id=None, dry_run=False):
        grant = GroupOrCIDR(self)
        grant.owner_id = owner_id
        grant.group_id = group_id
        grant.name = name
        grant.cidr_ip = cidr_ip
        self.grants.append(grant)
        return grant