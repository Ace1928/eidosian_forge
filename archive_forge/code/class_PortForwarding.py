from openstack import resource
class PortForwarding(resource.Resource):
    name_attribute = 'floating_ip_port_forwarding'
    resource_name = 'port forwarding'
    resource_key = 'port_forwarding'
    resources_key = 'port_forwardings'
    base_path = '/floatingips/%(floatingip_id)s/port_forwardings'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('internal_port_id', 'external_port', 'protocol', 'sort_key', 'sort_dir')
    floatingip_id = resource.URI('floatingip_id')
    internal_port_id = resource.Body('internal_port_id')
    internal_ip_address = resource.Body('internal_ip_address')
    internal_port = resource.Body('internal_port', type=int)
    external_port = resource.Body('external_port', type=int)
    protocol = resource.Body('protocol')
    description = resource.Body('description')