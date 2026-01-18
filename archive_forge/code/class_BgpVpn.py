from openstack import resource
class BgpVpn(resource.Resource):
    resource_key = 'bgpvpn'
    resources_key = 'bgpvpns'
    base_path = '/bgpvpn/bgpvpns'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    id = resource.Body('id')
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    route_distinguishers = resource.Body('route_distinguishers')
    route_targets = resource.Body('route_targets')
    import_targets = resource.Body('import_targets')
    export_targets = resource.Body('export_targets')
    local_pref = resource.Body('local_pref')
    vni = resource.Body('vni')
    type = resource.Body('type')