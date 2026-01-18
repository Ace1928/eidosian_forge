from boto.ec2.elb.healthcheck import HealthCheck
from boto.ec2.elb.listener import Listener
from boto.ec2.elb.listelement import ListElement
from boto.ec2.elb.policies import Policies, OtherPolicy
from boto.ec2.elb.securitygroup import SecurityGroup
from boto.ec2.instanceinfo import InstanceInfo
from boto.resultset import ResultSet
from boto.compat import six
class LoadBalancerZones(object):
    """
    Used to collect the zones for a Load Balancer when enable_zones
    or disable_zones are called.
    """

    def __init__(self, connection=None):
        self.connection = connection
        self.zones = ListElement()

    def startElement(self, name, attrs, connection):
        if name == 'AvailabilityZones':
            return self.zones

    def endElement(self, name, value, connection):
        pass