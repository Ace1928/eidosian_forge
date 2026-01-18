from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class DatasetAccessArray(object):

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
        return remove_nones_from_dict({u'domain': item.get('domain'), u'groupByEmail': item.get('group_by_email'), u'role': item.get('role'), u'specialGroup': item.get('special_group'), u'userByEmail': item.get('user_by_email'), u'view': DatasetView(item.get('view', {}), self.module).to_request()})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'domain': item.get(u'domain'), u'groupByEmail': item.get(u'groupByEmail'), u'role': item.get(u'role'), u'specialGroup': item.get(u'specialGroup'), u'userByEmail': item.get(u'userByEmail'), u'view': DatasetView(item.get(u'view', {}), self.module).from_response()})