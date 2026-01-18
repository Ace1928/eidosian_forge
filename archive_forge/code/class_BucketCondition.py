from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class BucketCondition(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'age': self.request.get('age_days'), u'createdBefore': self.request.get('created_before'), u'customTimeBefore': self.request.get('custom_time_before'), u'daysSinceCustomTime': self.request.get('days_since_custom_time'), u'daysSinceNoncurrentTime': self.request.get('days_since_noncurrent_time'), u'isLive': self.request.get('is_live'), u'matchesStorageClass': self.request.get('matches_storage_class'), u'noncurrentTimeBefore': self.request.get('noncurrent_time_before'), u'numNewerVersions': self.request.get('num_newer_versions')})

    def from_response(self):
        return remove_nones_from_dict({u'age': self.request.get(u'age'), u'createdBefore': self.request.get(u'createdBefore'), u'customTimeBefore': self.request.get(u'customTimeBefore'), u'daysSinceCustomTime': self.request.get(u'daysSinceCustomTime'), u'daysSinceNoncurrentTime': self.request.get(u'daysSinceNoncurrentTime'), u'isLive': self.request.get(u'isLive'), u'matchesStorageClass': self.request.get(u'matchesStorageClass'), u'noncurrentTimeBefore': self.request.get(u'noncurrentTimeBefore'), u'numNewerVersions': self.request.get(u'numNewerVersions')})