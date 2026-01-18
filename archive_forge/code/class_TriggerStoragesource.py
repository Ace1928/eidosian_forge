from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerStoragesource(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'bucket': self.request.get('bucket'), u'object': self.request.get('object'), u'generation': self.request.get('generation')})

    def from_response(self):
        return remove_nones_from_dict({u'bucket': self.request.get(u'bucket'), u'object': self.request.get(u'object'), u'generation': self.request.get(u'generation')})