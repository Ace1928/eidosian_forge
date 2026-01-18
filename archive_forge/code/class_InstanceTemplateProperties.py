from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceTemplateProperties(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'canIpForward': self.request.get('can_ip_forward'), u'description': self.request.get('description'), u'disks': InstanceTemplateDisksArray(self.request.get('disks', []), self.module).to_request(), u'labels': self.request.get('labels'), u'machineType': self.request.get('machine_type'), u'minCpuPlatform': self.request.get('min_cpu_platform'), u'metadata': self.request.get('metadata'), u'guestAccelerators': InstanceTemplateGuestacceleratorsArray(self.request.get('guest_accelerators', []), self.module).to_request(), u'networkInterfaces': InstanceTemplateNetworkinterfacesArray(self.request.get('network_interfaces', []), self.module).to_request(), u'scheduling': InstanceTemplateScheduling(self.request.get('scheduling', {}), self.module).to_request(), u'serviceAccounts': InstanceTemplateServiceaccountsArray(self.request.get('service_accounts', []), self.module).to_request(), u'tags': InstanceTemplateTags(self.request.get('tags', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'canIpForward': self.request.get(u'canIpForward'), u'description': self.request.get(u'description'), u'disks': InstanceTemplateDisksArray(self.request.get(u'disks', []), self.module).from_response(), u'labels': self.request.get(u'labels'), u'machineType': self.request.get(u'machineType'), u'minCpuPlatform': self.request.get(u'minCpuPlatform'), u'metadata': self.request.get(u'metadata'), u'guestAccelerators': InstanceTemplateGuestacceleratorsArray(self.request.get(u'guestAccelerators', []), self.module).from_response(), u'networkInterfaces': InstanceTemplateNetworkinterfacesArray(self.request.get(u'networkInterfaces', []), self.module).from_response(), u'scheduling': InstanceTemplateScheduling(self.request.get(u'scheduling', {}), self.module).from_response(), u'serviceAccounts': InstanceTemplateServiceaccountsArray(self.request.get(u'serviceAccounts', []), self.module).from_response(), u'tags': InstanceTemplateTags(self.request.get(u'tags', {}), self.module).from_response()})