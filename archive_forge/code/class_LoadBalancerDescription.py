from datetime import datetime
from boto.compat import six
class LoadBalancerDescription(BaseObject):

    def __init__(self, response):
        super(LoadBalancerDescription, self).__init__()
        self.domain = str(response['Domain'])
        self.listeners = []
        if response['Listeners']:
            for member in response['Listeners']:
                listener = Listener(member)
                self.listeners.append(listener)
        self.load_balancer_name = str(response['LoadBalancerName'])