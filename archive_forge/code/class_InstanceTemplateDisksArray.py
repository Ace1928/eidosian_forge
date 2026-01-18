from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceTemplateDisksArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({u'autoDelete': item.get('auto_delete'), u'boot': item.get('boot'), u'deviceName': item.get('device_name'), u'diskEncryptionKey': InstanceTemplateDiskencryptionkey(item.get('disk_encryption_key', {}), self.module).to_request(), u'index': item.get('index'), u'initializeParams': InstanceTemplateInitializeparams(item.get('initialize_params', {}), self.module).to_request(), u'interface': item.get('interface'), u'mode': item.get('mode'), u'source': replace_resource_dict(item.get(u'source', {}), 'name'), u'type': item.get('type')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'autoDelete': item.get(u'autoDelete'), u'boot': item.get(u'boot'), u'deviceName': item.get(u'deviceName'), u'diskEncryptionKey': InstanceTemplateDiskencryptionkey(item.get(u'diskEncryptionKey', {}), self.module).from_response(), u'index': item.get(u'index'), u'initializeParams': InstanceTemplateInitializeparams(self.module.params.get('initialize_params', {}), self.module).to_request(), u'interface': item.get(u'interface'), u'mode': item.get(u'mode'), u'source': item.get(u'source'), u'type': item.get(u'type')})