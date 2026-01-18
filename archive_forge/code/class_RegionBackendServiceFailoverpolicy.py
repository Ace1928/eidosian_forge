from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceFailoverpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'disableConnectionDrainOnFailover': self.request.get('disable_connection_drain_on_failover'), u'dropTrafficIfUnhealthy': self.request.get('drop_traffic_if_unhealthy'), u'failoverRatio': self.request.get('failover_ratio')})

    def from_response(self):
        return remove_nones_from_dict({u'disableConnectionDrainOnFailover': self.request.get(u'disableConnectionDrainOnFailover'), u'dropTrafficIfUnhealthy': self.request.get(u'dropTrafficIfUnhealthy'), u'failoverRatio': self.request.get(u'failoverRatio')})