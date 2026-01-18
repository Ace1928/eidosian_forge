from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicySchedule(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'hourlySchedule': ResourcePolicyHourlyschedule(self.request.get('hourly_schedule', {}), self.module).to_request(), u'dailySchedule': ResourcePolicyDailyschedule(self.request.get('daily_schedule', {}), self.module).to_request(), u'weeklySchedule': ResourcePolicyWeeklyschedule(self.request.get('weekly_schedule', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'hourlySchedule': ResourcePolicyHourlyschedule(self.request.get(u'hourlySchedule', {}), self.module).from_response(), u'dailySchedule': ResourcePolicyDailyschedule(self.request.get(u'dailySchedule', {}), self.module).from_response(), u'weeklySchedule': ResourcePolicyWeeklyschedule(self.request.get(u'weeklySchedule', {}), self.module).from_response()})