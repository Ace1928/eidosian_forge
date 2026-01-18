from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceTemplateInitializeparams(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'diskName': self.request.get('disk_name'), u'diskSizeGb': self.request.get('disk_size_gb'), u'diskType': disk_type_selflink(self.request.get('disk_type'), self.module.params), u'sourceImage': self.request.get('source_image'), u'sourceImageEncryptionKey': InstanceTemplateSourceimageencryptionkey(self.request.get('source_image_encryption_key', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'diskName': self.request.get(u'diskName'), u'diskSizeGb': self.request.get(u'diskSizeGb'), u'diskType': self.request.get(u'diskType'), u'sourceImage': self.request.get(u'sourceImage'), u'sourceImageEncryptionKey': InstanceTemplateSourceimageencryptionkey(self.request.get(u'sourceImageEncryptionKey', {}), self.module).from_response()})