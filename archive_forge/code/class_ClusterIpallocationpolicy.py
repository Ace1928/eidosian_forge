from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ClusterIpallocationpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'useIpAliases': self.request.get('use_ip_aliases'), u'createSubnetwork': self.request.get('create_subnetwork'), u'subnetworkName': self.request.get('subnetwork_name'), u'clusterSecondaryRangeName': self.request.get('cluster_secondary_range_name'), u'servicesSecondaryRangeName': self.request.get('services_secondary_range_name'), u'clusterIpv4CidrBlock': self.request.get('cluster_ipv4_cidr_block'), u'nodeIpv4CidrBlock': self.request.get('node_ipv4_cidr_block'), u'servicesIpv4CidrBlock': self.request.get('services_ipv4_cidr_block'), u'tpuIpv4CidrBlock': self.request.get('tpu_ipv4_cidr_block'), u'stackType': self.request.get('stack_type')})

    def from_response(self):
        return remove_nones_from_dict({u'useIpAliases': self.request.get(u'useIpAliases'), u'createSubnetwork': self.request.get(u'createSubnetwork'), u'subnetworkName': self.request.get(u'subnetworkName'), u'clusterSecondaryRangeName': self.request.get(u'clusterSecondaryRangeName'), u'servicesSecondaryRangeName': self.request.get(u'servicesSecondaryRangeName'), u'clusterIpv4CidrBlock': self.request.get(u'clusterIpv4CidrBlock'), u'nodeIpv4CidrBlock': self.request.get(u'nodeIpv4CidrBlock'), u'servicesIpv4CidrBlock': self.request.get(u'servicesIpv4CidrBlock'), u'tpuIpv4CidrBlock': self.request.get(u'tpuIpv4CidrBlock')})