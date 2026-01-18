from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapUrlrewrite(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'hostRewrite': self.request.get('host_rewrite'), u'pathPrefixRewrite': self.request.get('path_prefix_rewrite')})

    def from_response(self):
        return remove_nones_from_dict({u'hostRewrite': self.request.get(u'hostRewrite'), u'pathPrefixRewrite': self.request.get(u'pathPrefixRewrite')})