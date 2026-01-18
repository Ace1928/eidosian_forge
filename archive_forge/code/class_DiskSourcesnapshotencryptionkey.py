from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class DiskSourcesnapshotencryptionkey(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'rawKey': self.request.get('raw_key'), u'kmsKeyName': self.request.get('kms_key_name'), u'kmsKeyServiceAccount': self.request.get('kms_key_service_account')})

    def from_response(self):
        return remove_nones_from_dict({u'rawKey': self.request.get(u'rawKey'), u'kmsKeyName': self.request.get(u'kmsKeyName'), u'kmsKeyServiceAccount': self.request.get(u'kmsKeyServiceAccount')})