from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerArtifacts(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'images': self.request.get('images'), u'objects': TriggerObjects(self.request.get('objects', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'images': self.request.get(u'images'), u'objects': TriggerObjects(self.request.get(u'objects', {}), self.module).from_response()})