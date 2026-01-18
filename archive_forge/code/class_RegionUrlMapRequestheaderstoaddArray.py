from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapRequestheaderstoaddArray(object):

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
        return remove_nones_from_dict({u'headerName': item.get('header_name'), u'headerValue': item.get('header_value'), u'replace': item.get('replace')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'headerName': item.get(u'headerName'), u'headerValue': item.get(u'headerValue'), u'replace': item.get(u'replace')})