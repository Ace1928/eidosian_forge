from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceConsistenthash(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'httpCookie': BackendServiceHttpcookie(self.request.get('http_cookie', {}), self.module).to_request(), 'httpHeaderName': self.request.get('http_header_name'), 'minimumRingSize': self.request.get('minimum_ring_size')})

    def from_response(self):
        return remove_nones_from_dict({'httpCookie': BackendServiceHttpcookie(self.request.get('httpCookie', {}), self.module).from_response(), 'httpHeaderName': self.request.get('httpHeaderName'), 'minimumRingSize': self.request.get('minimumRingSize')})