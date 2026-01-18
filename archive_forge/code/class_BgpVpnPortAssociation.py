from openstack import resource
class BgpVpnPortAssociation(resource.Resource):
    resource_key = 'port_association'
    resources_key = 'port_associations'
    base_path = '/bgpvpn/bgpvpns/%(bgpvpn_id)s/port_associations'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    id = resource.Body('id')
    bgpvpn_id = resource.URI('bgpvpn_id')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    port_id = resource.Body('port_id')
    advertise_fixed_ips = resource.Body('advertise_fixed_ips')
    routes = resource.Body('routes')