from boto.ec2.elb.listelement import ListElement
from boto.resultset import ResultSet
from boto.ec2.autoscale.launchconfig import LaunchConfiguration
from boto.ec2.autoscale.request import Request
from boto.ec2.autoscale.instance import Instance
from boto.ec2.autoscale.tag import Tag
def _get_cooldown(self):
    return self.default_cooldown