from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class QueueAppengineroutingoverride(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'service': self.request.get('service'), u'version': self.request.get('version'), u'instance': self.request.get('instance')})

    def from_response(self):
        return remove_nones_from_dict({u'service': self.request.get(u'service'), u'version': self.request.get(u'version'), u'instance': self.request.get(u'instance')})