from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class NetworkAcl(TaggedEC2Object):

    def __init__(self, connection=None):
        super(NetworkAcl, self).__init__(connection)
        self.id = None
        self.vpc_id = None
        self.network_acl_entries = []
        self.associations = []

    def __repr__(self):
        return 'NetworkAcl:%s' % self.id

    def startElement(self, name, attrs, connection):
        result = super(NetworkAcl, self).startElement(name, attrs, connection)
        if result is not None:
            return result
        if name == 'entrySet':
            self.network_acl_entries = ResultSet([('item', NetworkAclEntry)])
            return self.network_acl_entries
        elif name == 'associationSet':
            self.associations = ResultSet([('item', NetworkAclAssociation)])
            return self.associations
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'networkAclId':
            self.id = value
        elif name == 'vpcId':
            self.vpc_id = value
        else:
            setattr(self, name, value)