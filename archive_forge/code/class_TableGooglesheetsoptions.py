from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableGooglesheetsoptions(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'skipLeadingRows': self.request.get('skip_leading_rows')})

    def from_response(self):
        return remove_nones_from_dict({u'skipLeadingRows': self.request.get(u'skipLeadingRows')})