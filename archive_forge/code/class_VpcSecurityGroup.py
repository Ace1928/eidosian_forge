from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.resultset import ResultSet
class VpcSecurityGroup(object):
    """
    Describes a VPC security group for use in a OptionGroup
    """

    def __init__(self, vpc_id=None, status=None):
        self.vpc_id = vpc_id
        self.status = status

    def __repr__(self):
        return 'VpcSecurityGroup:%s' % self.vpc_id

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'VpcSecurityGroupId':
            self.vpc_id = value
        elif name == 'Status':
            self.status = value
        else:
            setattr(self, name, value)