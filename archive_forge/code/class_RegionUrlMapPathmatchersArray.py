from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapPathmatchersArray(object):

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
        return remove_nones_from_dict({u'defaultService': replace_resource_dict(item.get(u'default_service', {}), 'selfLink'), u'description': item.get('description'), u'name': item.get('name'), u'routeRules': RegionUrlMapRouterulesArray(item.get('route_rules', []), self.module).to_request(), u'pathRules': RegionUrlMapPathrulesArray(item.get('path_rules', []), self.module).to_request(), u'defaultUrlRedirect': RegionUrlMapDefaulturlredirect(item.get('default_url_redirect', {}), self.module).to_request()})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'defaultService': item.get(u'defaultService'), u'description': item.get(u'description'), u'name': item.get(u'name'), u'routeRules': RegionUrlMapRouterulesArray(item.get(u'routeRules', []), self.module).from_response(), u'pathRules': RegionUrlMapPathrulesArray(item.get(u'pathRules', []), self.module).from_response(), u'defaultUrlRedirect': RegionUrlMapDefaulturlredirect(item.get(u'defaultUrlRedirect', {}), self.module).from_response()})