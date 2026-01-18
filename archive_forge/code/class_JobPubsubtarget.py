from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class JobPubsubtarget(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'topicName': self.request.get('topic_name'), u'data': self.request.get('data'), u'attributes': self.request.get('attributes')})

    def from_response(self):
        return remove_nones_from_dict({u'topicName': self.request.get(u'topicName'), u'data': self.request.get(u'data'), u'attributes': self.request.get(u'attributes')})