from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableFieldsArray(object):

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
        return remove_nones_from_dict({u'description': item.get('description'), u'fields': item.get('fields'), u'mode': item.get('mode'), u'name': item.get('name'), u'type': item.get('type')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'description': item.get(u'description'), u'fields': item.get(u'fields'), u'mode': item.get(u'mode'), u'name': item.get(u'name'), u'type': item.get(u'type')})