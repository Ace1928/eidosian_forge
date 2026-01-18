from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicyInstanceschedulepolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'vmStartSchedule': ResourcePolicyVmstartschedule(self.request.get('vm_start_schedule', {}), self.module).to_request(), u'vmStopSchedule': ResourcePolicyVmstopschedule(self.request.get('vm_stop_schedule', {}), self.module).to_request(), u'timeZone': self.request.get('time_zone'), u'startTime': self.request.get('start_time'), u'expirationTime': self.request.get('expiration_time')})

    def from_response(self):
        return remove_nones_from_dict({u'vmStartSchedule': ResourcePolicyVmstartschedule(self.request.get(u'vmStartSchedule', {}), self.module).from_response(), u'vmStopSchedule': ResourcePolicyVmstopschedule(self.request.get(u'vmStopSchedule', {}), self.module).from_response(), u'timeZone': self.request.get(u'timeZone'), u'startTime': self.request.get(u'startTime'), u'expirationTime': self.request.get(u'expirationTime')})