from openstack.common import tag
from openstack.network.v2 import _base
from openstack import resource
class DHCPAgentHostingNetwork(Network):
    resource_key = 'network'
    resources_key = 'networks'
    base_path = '/agents/%(agent_id)s/dhcp-networks'
    resource_name = 'dhcp-network'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True