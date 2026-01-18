from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
class AutoScalingGroupMetric(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.metric = None
        self.granularity = None

    def __repr__(self):
        return 'AutoScalingGroupMetric:%s' % self.metric

    def startElement(self, name, attrs, connection):
        return

    def endElement(self, name, value, connection):
        if name == 'Metric':
            self.metric = value
        elif name == 'Granularity':
            self.granularity = value
        else:
            setattr(self, name, value)