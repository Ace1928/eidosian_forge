from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceTemplateNetworkinterfacesArray(object):

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
        return remove_nones_from_dict({u'accessConfigs': InstanceTemplateAccessconfigsArray(item.get('access_configs', []), self.module).to_request(), u'aliasIpRanges': InstanceTemplateAliasiprangesArray(item.get('alias_ip_ranges', []), self.module).to_request(), u'network': replace_resource_dict(item.get(u'network', {}), 'selfLink'), u'networkIP': item.get('network_ip'), u'subnetwork': replace_resource_dict(item.get(u'subnetwork', {}), 'selfLink')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'accessConfigs': InstanceTemplateAccessconfigsArray(item.get(u'accessConfigs', []), self.module).from_response(), u'aliasIpRanges': InstanceTemplateAliasiprangesArray(item.get(u'aliasIpRanges', []), self.module).from_response(), u'network': item.get(u'network'), u'networkIP': item.get(u'networkIP'), u'subnetwork': item.get(u'subnetwork')})