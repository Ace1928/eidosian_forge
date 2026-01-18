from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceAccessconfigsArray(object):

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
        return remove_nones_from_dict({u'name': item.get('name'), u'natIP': replace_resource_dict(item.get(u'nat_ip', {}), 'address'), u'type': item.get('type'), u'setPublicPtr': item.get('set_public_ptr'), u'publicPtrDomainName': item.get('public_ptr_domain_name'), u'networkTier': item.get('network_tier')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'name': item.get(u'name'), u'natIP': item.get(u'natIP'), u'type': item.get(u'type'), u'setPublicPtr': item.get(u'setPublicPtr'), u'publicPtrDomainName': item.get(u'publicPtrDomainName'), u'networkTier': item.get(u'networkTier')})