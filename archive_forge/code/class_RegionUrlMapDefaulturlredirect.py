from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapDefaulturlredirect(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'hostRedirect': self.request.get('host_redirect'), u'httpsRedirect': self.request.get('https_redirect'), u'pathRedirect': self.request.get('path_redirect'), u'prefixRedirect': self.request.get('prefix_redirect'), u'redirectResponseCode': self.request.get('redirect_response_code'), u'stripQuery': self.request.get('strip_query')})

    def from_response(self):
        return remove_nones_from_dict({u'hostRedirect': self.request.get(u'hostRedirect'), u'httpsRedirect': self.request.get(u'httpsRedirect'), u'pathRedirect': self.request.get(u'pathRedirect'), u'prefixRedirect': self.request.get(u'prefixRedirect'), u'redirectResponseCode': self.request.get(u'redirectResponseCode'), u'stripQuery': self.request.get(u'stripQuery')})