from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class ImageRawdisk(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'containerType': self.request.get('container_type'), u'sha1Checksum': self.request.get('sha1_checksum'), u'source': self.request.get('source')})

    def from_response(self):
        return remove_nones_from_dict({u'containerType': self.request.get(u'containerType'), u'sha1Checksum': self.request.get(u'sha1Checksum'), u'source': self.request.get(u'source')})