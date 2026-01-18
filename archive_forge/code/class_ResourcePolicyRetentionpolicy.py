from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicyRetentionpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'maxRetentionDays': self.request.get('max_retention_days'), u'onSourceDiskDelete': self.request.get('on_source_disk_delete')})

    def from_response(self):
        return remove_nones_from_dict({u'maxRetentionDays': self.request.get(u'maxRetentionDays'), u'onSourceDiskDelete': self.request.get(u'onSourceDiskDelete')})