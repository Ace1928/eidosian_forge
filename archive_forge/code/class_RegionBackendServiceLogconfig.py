from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceLogconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enable': self.request.get('enable'), u'sampleRate': self.request.get('sample_rate')})

    def from_response(self):
        return remove_nones_from_dict({u'enable': self.request.get(u'enable'), u'sampleRate': self.request.get(u'sampleRate')})