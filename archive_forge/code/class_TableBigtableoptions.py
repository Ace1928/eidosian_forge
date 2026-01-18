from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableBigtableoptions(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'ignoreUnspecifiedColumnFamilies': self.request.get('ignore_unspecified_column_families'), u'readRowkeyAsString': self.request.get('read_rowkey_as_string'), u'columnFamilies': TableColumnfamiliesArray(self.request.get('column_families', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'ignoreUnspecifiedColumnFamilies': self.request.get(u'ignoreUnspecifiedColumnFamilies'), u'readRowkeyAsString': self.request.get(u'readRowkeyAsString'), u'columnFamilies': TableColumnfamiliesArray(self.request.get(u'columnFamilies', []), self.module).from_response()})