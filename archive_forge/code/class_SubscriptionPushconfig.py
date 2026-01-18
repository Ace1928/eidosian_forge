from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class SubscriptionPushconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'oidcToken': SubscriptionOidctoken(self.request.get('oidc_token', {}), self.module).to_request(), u'pushEndpoint': self.request.get('push_endpoint'), u'attributes': self.request.get('attributes')})

    def from_response(self):
        return remove_nones_from_dict({u'oidcToken': SubscriptionOidctoken(self.request.get(u'oidcToken', {}), self.module).from_response(), u'pushEndpoint': self.request.get(u'pushEndpoint'), u'attributes': self.request.get(u'attributes')})