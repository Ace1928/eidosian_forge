from datetime import datetime
from boto.compat import six
class EnvironmentResourcesDescription(BaseObject):

    def __init__(self, response):
        super(EnvironmentResourcesDescription, self).__init__()
        if response['LoadBalancer']:
            self.load_balancer = LoadBalancerDescription(response['LoadBalancer'])
        else:
            self.load_balancer = None