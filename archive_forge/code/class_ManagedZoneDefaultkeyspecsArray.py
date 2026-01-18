from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class ManagedZoneDefaultkeyspecsArray(object):

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
        return remove_nones_from_dict({u'algorithm': item.get('algorithm'), u'keyLength': item.get('key_length'), u'keyType': item.get('key_type'), u'kind': item.get('kind')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'algorithm': item.get(u'algorithm'), u'keyLength': item.get(u'keyLength'), u'keyType': item.get(u'keyType'), u'kind': item.get(u'kind')})