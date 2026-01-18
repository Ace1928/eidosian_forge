from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapDelay(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'fixedDelay': UrlMapFixeddelay(self.request.get('fixed_delay', {}), self.module).to_request(), u'percentage': self.request.get('percentage')})

    def from_response(self):
        return remove_nones_from_dict({u'fixedDelay': UrlMapFixeddelay(self.request.get(u'fixedDelay', {}), self.module).from_response(), u'percentage': self.request.get(u'percentage')})