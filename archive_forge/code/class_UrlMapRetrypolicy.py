from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapRetrypolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'retryConditions': self.request.get('retry_conditions'), u'numRetries': self.request.get('num_retries'), u'perTryTimeout': UrlMapPertrytimeout(self.request.get('per_try_timeout', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'retryConditions': self.request.get(u'retryConditions'), u'numRetries': self.request.get(u'numRetries'), u'perTryTimeout': UrlMapPertrytimeout(self.request.get(u'perTryTimeout', {}), self.module).from_response()})