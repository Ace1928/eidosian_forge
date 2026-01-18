from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class QueueRatelimits(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'maxDispatchesPerSecond': self.request.get('max_dispatches_per_second'), u'maxConcurrentDispatches': self.request.get('max_concurrent_dispatches')})

    def from_response(self):
        return remove_nones_from_dict({u'maxDispatchesPerSecond': self.request.get(u'maxDispatchesPerSecond'), u'maxConcurrentDispatches': self.request.get(u'maxConcurrentDispatches')})