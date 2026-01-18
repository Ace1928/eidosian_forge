from boto.ec2.ec2object import TaggedEC2Object
from boto.resultset import ResultSet
class RouteAssociation(object):

    def __init__(self, connection=None):
        self.id = None
        self.route_table_id = None
        self.subnet_id = None
        self.main = False

    def __repr__(self):
        return 'RouteAssociation:%s' % self.id

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'routeTableAssociationId':
            self.id = value
        elif name == 'routeTableId':
            self.route_table_id = value
        elif name == 'subnetId':
            self.subnet_id = value
        elif name == 'main':
            self.main = value == 'true'