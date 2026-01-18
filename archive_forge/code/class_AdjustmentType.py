from boto.resultset import ResultSet
from boto.ec2.elb.listelement import ListElement
class AdjustmentType(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.adjustment_type = None

    def __repr__(self):
        return 'AdjustmentType:%s' % self.adjustment_type

    def startElement(self, name, attrs, connection):
        return

    def endElement(self, name, value, connection):
        if name == 'AdjustmentType':
            self.adjustment_type = value
        return