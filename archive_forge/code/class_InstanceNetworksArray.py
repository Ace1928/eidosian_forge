from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceNetworksArray(object):

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
        return remove_nones_from_dict({u'network': item.get('network'), u'modes': item.get('modes'), u'reservedIpRange': item.get('reserved_ip_range')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'network': self.module.params.get('network'), u'modes': self.module.params.get('modes'), u'reservedIpRange': item.get(u'reservedIpRange')})