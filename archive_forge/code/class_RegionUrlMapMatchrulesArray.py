from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapMatchrulesArray(object):

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
        return remove_nones_from_dict({u'fullPathMatch': item.get('full_path_match'), u'headerMatches': RegionUrlMapHeadermatchesArray(item.get('header_matches', []), self.module).to_request(), u'ignoreCase': item.get('ignore_case'), u'metadataFilters': RegionUrlMapMetadatafiltersArray(item.get('metadata_filters', []), self.module).to_request(), u'prefixMatch': item.get('prefix_match'), u'queryParameterMatches': RegionUrlMapQueryparametermatchesArray(item.get('query_parameter_matches', []), self.module).to_request(), u'regexMatch': item.get('regex_match')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'fullPathMatch': item.get(u'fullPathMatch'), u'headerMatches': RegionUrlMapHeadermatchesArray(item.get(u'headerMatches', []), self.module).from_response(), u'ignoreCase': item.get(u'ignoreCase'), u'metadataFilters': RegionUrlMapMetadatafiltersArray(item.get(u'metadataFilters', []), self.module).from_response(), u'prefixMatch': item.get(u'prefixMatch'), u'queryParameterMatches': RegionUrlMapQueryparametermatchesArray(item.get(u'queryParameterMatches', []), self.module).from_response(), u'regexMatch': item.get(u'regexMatch')})