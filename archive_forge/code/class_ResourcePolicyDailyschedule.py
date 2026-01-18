from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicyDailyschedule(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'daysInCycle': self.request.get('days_in_cycle'), u'startTime': self.request.get('start_time')})

    def from_response(self):
        return remove_nones_from_dict({u'daysInCycle': self.request.get(u'daysInCycle'), u'startTime': self.request.get(u'startTime')})