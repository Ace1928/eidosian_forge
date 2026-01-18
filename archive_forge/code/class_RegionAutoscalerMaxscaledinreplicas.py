from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionAutoscalerMaxscaledinreplicas(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'fixed': self.request.get('fixed'), u'percent': self.request.get('percent')})

    def from_response(self):
        return remove_nones_from_dict({u'fixed': self.request.get(u'fixed'), u'percent': self.request.get(u'percent')})