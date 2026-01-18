from openstack import resource
class LocalIPAssociation(resource.Resource):
    """Local IP extension."""
    resource_key = 'port_association'
    resources_key = 'port_associations'
    base_path = '/local_ips/%(local_ip_id)s/port_associations'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _allow_unknown_attrs_in_body = True
    _query_mapping = resource.QueryParameters('fixed_port_id', 'fixed_ip', 'host', 'sort_key', 'sort_dir')
    fixed_port_id = resource.Body('fixed_port_id')
    fixed_ip = resource.Body('fixed_ip')
    host = resource.Body('host')
    local_ip_address = resource.Body('local_ip_address')
    local_ip_id = resource.URI('local_ip_id')