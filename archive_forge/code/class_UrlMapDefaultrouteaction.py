from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapDefaultrouteaction(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'weightedBackendServices': UrlMapWeightedbackendservicesArray(self.request.get('weighted_backend_services', []), self.module).to_request(), u'urlRewrite': UrlMapUrlrewrite(self.request.get('url_rewrite', {}), self.module).to_request(), u'timeout': UrlMapTimeout(self.request.get('timeout', {}), self.module).to_request(), u'retryPolicy': UrlMapRetrypolicy(self.request.get('retry_policy', {}), self.module).to_request(), u'requestMirrorPolicy': UrlMapRequestmirrorpolicy(self.request.get('request_mirror_policy', {}), self.module).to_request(), u'corsPolicy': UrlMapCorspolicy(self.request.get('cors_policy', {}), self.module).to_request(), u'faultInjectionPolicy': UrlMapFaultinjectionpolicy(self.request.get('fault_injection_policy', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'weightedBackendServices': UrlMapWeightedbackendservicesArray(self.request.get(u'weightedBackendServices', []), self.module).from_response(), u'urlRewrite': UrlMapUrlrewrite(self.request.get(u'urlRewrite', {}), self.module).from_response(), u'timeout': UrlMapTimeout(self.request.get(u'timeout', {}), self.module).from_response(), u'retryPolicy': UrlMapRetrypolicy(self.request.get(u'retryPolicy', {}), self.module).from_response(), u'requestMirrorPolicy': UrlMapRequestmirrorpolicy(self.request.get(u'requestMirrorPolicy', {}), self.module).from_response(), u'corsPolicy': UrlMapCorspolicy(self.request.get(u'corsPolicy', {}), self.module).from_response(), u'faultInjectionPolicy': UrlMapFaultinjectionpolicy(self.request.get(u'faultInjectionPolicy', {}), self.module).from_response()})