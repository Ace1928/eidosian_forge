from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ClusterPrivateclusterconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enablePrivateNodes': self.request.get('enable_private_nodes'), u'enablePrivateEndpoint': self.request.get('enable_private_endpoint'), u'masterIpv4CidrBlock': self.request.get('master_ipv4_cidr_block')})

    def from_response(self):
        return remove_nones_from_dict({u'enablePrivateNodes': self.request.get(u'enablePrivateNodes'), u'enablePrivateEndpoint': self.request.get(u'enablePrivateEndpoint'), u'masterIpv4CidrBlock': self.request.get(u'masterIpv4CidrBlock')})