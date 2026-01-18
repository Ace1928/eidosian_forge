from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapCorspolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'allowOrigins': self.request.get('allow_origins'), u'allowOriginRegexes': self.request.get('allow_origin_regexes'), u'allowMethods': self.request.get('allow_methods'), u'allowHeaders': self.request.get('allow_headers'), u'exposeHeaders': self.request.get('expose_headers'), u'maxAge': self.request.get('max_age'), u'allowCredentials': self.request.get('allow_credentials'), u'disabled': self.request.get('disabled')})

    def from_response(self):
        return remove_nones_from_dict({u'allowOrigins': self.request.get(u'allowOrigins'), u'allowOriginRegexes': self.request.get(u'allowOriginRegexes'), u'allowMethods': self.request.get(u'allowMethods'), u'allowHeaders': self.request.get(u'allowHeaders'), u'exposeHeaders': self.request.get(u'exposeHeaders'), u'maxAge': self.request.get(u'maxAge'), u'allowCredentials': self.request.get(u'allowCredentials'), u'disabled': self.request.get(u'disabled')})