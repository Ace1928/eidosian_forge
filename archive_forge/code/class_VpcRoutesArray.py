from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (Config, HwcClientException,
import re
class VpcRoutesArray(object):

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
        return {u'destination': item.get('destination'), u'nexthop': item.get('next_hop')}

    def _response_from_item(self, item):
        return {u'destination': item.get(u'destination'), u'next_hop': item.get(u'nexthop')}