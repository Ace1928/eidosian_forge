from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceHttpcookie(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'ttl': BackendServiceTtl(self.request.get('ttl', {}), self.module).to_request(), 'name': self.request.get('name'), 'path': self.request.get('path')})

    def from_response(self):
        return remove_nones_from_dict({'ttl': BackendServiceTtl(self.request.get('ttl', {}), self.module).from_response(), 'name': self.request.get('name'), 'path': self.request.get('path')})