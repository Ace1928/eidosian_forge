from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class CryptoKeyVersiontemplate(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'algorithm': self.request.get('algorithm'), u'protectionLevel': self.request.get('protection_level')})

    def from_response(self):
        return remove_nones_from_dict({u'algorithm': self.request.get(u'algorithm'), u'protectionLevel': self.module.params.get('protection_level')})