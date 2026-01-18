from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceShieldedinstanceconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enableSecureBoot': self.request.get('enable_secure_boot'), u'enableVtpm': self.request.get('enable_vtpm'), u'enableIntegrityMonitoring': self.request.get('enable_integrity_monitoring')})

    def from_response(self):
        return remove_nones_from_dict({u'enableSecureBoot': self.request.get(u'enableSecureBoot'), u'enableVtpm': self.request.get(u'enableVtpm'), u'enableIntegrityMonitoring': self.request.get(u'enableIntegrityMonitoring')})