from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapRouterulesArray(object):

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
        return remove_nones_from_dict({u'priority': item.get('priority'), u'service': replace_resource_dict(item.get(u'service', {}), 'selfLink'), u'headerAction': UrlMapHeaderaction(item.get('header_action', {}), self.module).to_request(), u'matchRules': UrlMapMatchrulesArray(item.get('match_rules', []), self.module).to_request(), u'routeAction': UrlMapRouteaction(item.get('route_action', {}), self.module).to_request(), u'urlRedirect': UrlMapUrlredirect(item.get('url_redirect', {}), self.module).to_request()})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'priority': item.get(u'priority'), u'service': item.get(u'service'), u'headerAction': UrlMapHeaderaction(item.get(u'headerAction', {}), self.module).from_response(), u'matchRules': UrlMapMatchrulesArray(item.get(u'matchRules', []), self.module).from_response(), u'routeAction': UrlMapRouteaction(item.get(u'routeAction', {}), self.module).from_response(), u'urlRedirect': UrlMapUrlredirect(item.get(u'urlRedirect', {}), self.module).from_response()})