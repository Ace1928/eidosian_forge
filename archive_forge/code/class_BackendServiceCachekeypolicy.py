from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceCachekeypolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'includeHost': self.request.get('include_host'), 'includeProtocol': self.request.get('include_protocol'), 'includeQueryString': self.request.get('include_query_string'), 'queryStringBlacklist': self.request.get('query_string_blacklist'), 'queryStringWhitelist': self.request.get('query_string_whitelist')})

    def from_response(self):
        return remove_nones_from_dict({'includeHost': self.request.get('includeHost'), 'includeProtocol': self.request.get('includeProtocol'), 'includeQueryString': self.request.get('includeQueryString'), 'queryStringBlacklist': self.request.get('queryStringBlacklist'), 'queryStringWhitelist': self.request.get('queryStringWhitelist')})