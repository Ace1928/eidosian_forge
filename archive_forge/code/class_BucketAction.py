from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class BucketAction(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'storageClass': self.request.get('storage_class'), u'type': self.request.get('type')})

    def from_response(self):
        return remove_nones_from_dict({u'storageClass': self.request.get(u'storageClass'), u'type': self.request.get(u'type')})