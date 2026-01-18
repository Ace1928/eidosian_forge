from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ClusterNetworkconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enableIntraNodeVisibility': self.request.get('enable_intra_node_visibility'), u'defaultSnatStatus': self.request.get('default_snat_status'), u'datapathProvider': self.request.get('datapath_provider')})

    def from_response(self):
        return remove_nones_from_dict({u'enableIntraNodeVisibility': self.request.get(u'enableIntraNodeVisibility'), u'defaultSnatStatus': self.request.get(u'defaultSnatStatus'), u'datapathProvider': self.request.get('datapath_provider')})