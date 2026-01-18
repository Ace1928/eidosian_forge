from openstack import resource
class VpnEndpointGroup(resource.Resource):
    resource_key = 'endpoint_group'
    resources_key = 'endpoint_groups'
    base_path = '/vpn/endpoint-groups'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'tenant_id', type='endpoint_type')
    description = resource.Body('description')
    endpoints = resource.Body('endpoints', type=list)
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    type = resource.Body('type')