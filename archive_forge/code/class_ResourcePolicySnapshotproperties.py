from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicySnapshotproperties(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'labels': self.request.get('labels'), u'storageLocations': self.request.get('storage_locations'), u'guestFlush': self.request.get('guest_flush')})

    def from_response(self):
        return remove_nones_from_dict({u'labels': self.request.get(u'labels'), u'storageLocations': self.request.get(u'storageLocations'), u'guestFlush': self.request.get(u'guestFlush')})