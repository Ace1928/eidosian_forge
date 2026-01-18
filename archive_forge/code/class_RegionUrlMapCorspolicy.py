from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapCorspolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'allowCredentials': self.request.get('allow_credentials'), u'allowHeaders': self.request.get('allow_headers'), u'allowMethods': self.request.get('allow_methods'), u'allowOriginRegexes': self.request.get('allow_origin_regexes'), u'allowOrigins': self.request.get('allow_origins'), u'disabled': self.request.get('disabled'), u'exposeHeaders': self.request.get('expose_headers'), u'maxAge': self.request.get('max_age')})

    def from_response(self):
        return remove_nones_from_dict({u'allowCredentials': self.request.get(u'allowCredentials'), u'allowHeaders': self.request.get(u'allowHeaders'), u'allowMethods': self.request.get(u'allowMethods'), u'allowOriginRegexes': self.request.get(u'allowOriginRegexes'), u'allowOrigins': self.request.get(u'allowOrigins'), u'disabled': self.request.get(u'disabled'), u'exposeHeaders': self.request.get(u'exposeHeaders'), u'maxAge': self.request.get(u'maxAge')})