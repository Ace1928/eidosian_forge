from boto.ec2.ec2object import TaggedEC2Object
class VpcInfo(object):

    def __init__(self):
        """
        Information on peer Vpc.
        
        :ivar id: The unique ID of peer Vpc.
        :ivar owner_id: Owner of peer Vpc.
        :ivar cidr_block: CIDR Block of peer Vpc.
        """
        self.vpc_id = None
        self.owner_id = None
        self.cidr_block = None

    def __repr__(self):
        return 'VpcInfo:%s' % self.vpc_id

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'vpcId':
            self.vpc_id = value
        elif name == 'ownerId':
            self.owner_id = value
        elif name == 'cidrBlock':
            self.cidr_block = value
        else:
            setattr(self, name, value)