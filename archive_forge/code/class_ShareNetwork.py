from openstack import resource
from openstack.shared_file_system.v2 import share_network_subnet
class ShareNetwork(resource.Resource):
    resource_key = 'share_network'
    resources_key = 'share_networks'
    base_path = '/share-networks'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_head = False
    _query_mapping = resource.QueryParameters('project_id', 'name', 'description', 'created_since', 'created_before', 'security_service_id', 'limit', 'offset', all_projects='all_tenants')
    created_at = resource.Body('created_at')
    description = resource.Body('description', type=str)
    project_id = resource.Body('project_id', type=str)
    share_network_subnets = resource.Body('share_network_subnets', type=list, list_type=share_network_subnet.ShareNetworkSubnet)
    neutron_net_id = resource.Body('neutron_net_id', type=str)
    neutron_subnet_id = resource.Body('neutron_subnet_id', type=str)
    updated_at = resource.Body('updated_at', type=str)