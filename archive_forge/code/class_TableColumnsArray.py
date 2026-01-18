from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableColumnsArray(object):

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
        return remove_nones_from_dict({u'encoding': item.get('encoding'), u'fieldName': item.get('field_name'), u'onlyReadLatest': item.get('only_read_latest'), u'qualifierString': item.get('qualifier_string'), u'type': item.get('type')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'encoding': item.get(u'encoding'), u'fieldName': item.get(u'fieldName'), u'onlyReadLatest': item.get(u'onlyReadLatest'), u'qualifierString': item.get(u'qualifierString'), u'type': item.get(u'type')})