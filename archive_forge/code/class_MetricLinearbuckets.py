from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class MetricLinearbuckets(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'numFiniteBuckets': self.request.get('num_finite_buckets'), u'width': self.request.get('width'), u'offset': self.request.get('offset')})

    def from_response(self):
        return remove_nones_from_dict({u'numFiniteBuckets': self.request.get(u'numFiniteBuckets'), u'width': self.request.get(u'width'), u'offset': self.request.get(u'offset')})