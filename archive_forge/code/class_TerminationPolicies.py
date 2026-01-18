from boto.resultset import ResultSet
from boto.ec2.elb.listelement import ListElement
class TerminationPolicies(list):

    def __init__(self, connection=None, **kwargs):
        pass

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'member':
            self.append(value)