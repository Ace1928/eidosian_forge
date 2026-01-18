from openstack.network.v2 import _base
from openstack import resource
class DefaultSecurityGroupRule(_base.NetworkResource):
    resource_key = 'default_security_group_rule'
    resources_key = 'default_security_group_rules'
    base_path = '/default-security-group-rules'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('id', 'description', 'remote_group_id', 'remote_address_group_id', 'direction', 'protocol', 'port_range_min', 'port_range_max', 'remote_ip_prefix', 'used_in_default_sg', 'used_in_non_default_sg', 'sort_dir', 'sort_key', ether_type='ethertype')
    description = resource.Body('description')
    remote_group_id = resource.Body('remote_group_id')
    remote_address_group_id = resource.Body('remote_address_group_id')
    direction = resource.Body('direction')
    protocol = resource.Body('protocol')
    port_range_min = resource.Body('port_range_min', type=int)
    port_range_max = resource.Body('port_range_max', type=int)
    remote_ip_prefix = resource.Body('remote_ip_prefix')
    ether_type = resource.Body('ethertype')
    used_in_default_sg = resource.Body('used_in_default_sg', type=bool)
    used_in_non_default_sg = resource.Body('used_in_non_default_sg', type=bool)