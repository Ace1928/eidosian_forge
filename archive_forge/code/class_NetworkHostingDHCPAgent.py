from openstack import exceptions
from openstack import resource
from openstack import utils
class NetworkHostingDHCPAgent(Agent):
    resource_key = 'agent'
    resources_key = 'agents'
    resource_name = 'dhcp-agent'
    base_path = '/networks/%(network_id)s/dhcp-agents'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True