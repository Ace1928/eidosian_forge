from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class JobHttptarget(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'uri': self.request.get('uri'), u'httpMethod': self.request.get('http_method'), u'body': self.request.get('body'), u'headers': self.request.get('headers'), u'oauthToken': JobOauthtoken(self.request.get('oauth_token', {}), self.module).to_request(), u'oidcToken': JobOidctoken(self.request.get('oidc_token', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'uri': self.request.get(u'uri'), u'httpMethod': self.request.get(u'httpMethod'), u'body': self.request.get(u'body'), u'headers': self.request.get(u'headers'), u'oauthToken': JobOauthtoken(self.request.get(u'oauthToken', {}), self.module).from_response(), u'oidcToken': JobOidctoken(self.request.get(u'oidcToken', {}), self.module).from_response()})