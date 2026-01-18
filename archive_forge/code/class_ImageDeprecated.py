from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class ImageDeprecated(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'deleted': self.request.get('deleted'), u'deprecated': self.request.get('deprecated'), u'obsolete': self.request.get('obsolete'), u'replacement': self.request.get('replacement'), u'state': self.request.get('state')})

    def from_response(self):
        return remove_nones_from_dict({u'deleted': self.request.get(u'deleted'), u'deprecated': self.request.get(u'deprecated'), u'obsolete': self.request.get(u'obsolete'), u'replacement': self.request.get(u'replacement'), u'state': self.request.get(u'state')})