from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class BucketCorsArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({u'maxAgeSeconds': item.get('max_age_seconds'), u'method': item.get('method'), u'origin': item.get('origin'), u'responseHeader': item.get('response_header')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'maxAgeSeconds': item.get(u'maxAgeSeconds'), u'method': item.get(u'method'), u'origin': item.get(u'origin'), u'responseHeader': item.get(u'responseHeader')})