from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapTestsArray(object):

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
        return remove_nones_from_dict({u'description': item.get('description'), u'host': item.get('host'), u'path': item.get('path'), u'service': replace_resource_dict(item.get(u'service', {}), 'selfLink')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'description': item.get(u'description'), u'host': item.get(u'host'), u'path': item.get(u'path'), u'service': item.get(u'service')})