from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class SubscriptionOidctoken(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'serviceAccountEmail': self.request.get('service_account_email'), u'audience': self.request.get('audience')})

    def from_response(self):
        return remove_nones_from_dict({u'serviceAccountEmail': self.request.get(u'serviceAccountEmail'), u'audience': self.request.get(u'audience')})