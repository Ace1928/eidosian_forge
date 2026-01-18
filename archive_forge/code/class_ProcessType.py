from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
class ProcessType(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.process_name = None

    def __repr__(self):
        return 'ProcessType(%s)' % self.process_name

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'ProcessName':
            self.process_name = value