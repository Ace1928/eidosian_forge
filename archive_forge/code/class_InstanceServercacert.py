from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class InstanceServercacert(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'cert': self.request.get('cert'), u'certSerialNumber': self.request.get('cert_serial_number'), u'commonName': self.request.get('common_name'), u'createTime': self.request.get('create_time'), u'expirationTime': self.request.get('expiration_time'), u'sha1Fingerprint': self.request.get('sha1_fingerprint')})

    def from_response(self):
        return remove_nones_from_dict({u'cert': self.request.get(u'cert'), u'certSerialNumber': self.request.get(u'certSerialNumber'), u'commonName': self.request.get(u'commonName'), u'createTime': self.request.get(u'createTime'), u'expirationTime': self.request.get(u'expirationTime'), u'sha1Fingerprint': self.request.get(u'sha1Fingerprint')})