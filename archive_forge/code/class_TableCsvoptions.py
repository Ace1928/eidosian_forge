from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableCsvoptions(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'allowJaggedRows': self.request.get('allow_jagged_rows'), u'allowQuotedNewlines': self.request.get('allow_quoted_newlines'), u'encoding': self.request.get('encoding'), u'fieldDelimiter': self.request.get('field_delimiter'), u'quote': self.request.get('quote'), u'skipLeadingRows': self.request.get('skip_leading_rows')})

    def from_response(self):
        return remove_nones_from_dict({u'allowJaggedRows': self.request.get(u'allowJaggedRows'), u'allowQuotedNewlines': self.request.get(u'allowQuotedNewlines'), u'encoding': self.request.get(u'encoding'), u'fieldDelimiter': self.request.get(u'fieldDelimiter'), u'quote': self.request.get(u'quote'), u'skipLeadingRows': self.request.get(u'skipLeadingRows')})