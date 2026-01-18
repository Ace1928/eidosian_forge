from openstack import resource
class PoolMember(resource.Resource):
    resource_key = 'member'
    resources_key = 'members'
    base_path = '/lbaas/pools/%(pool_id)s/members'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('address', 'name', 'protocol_port', 'subnet_id', 'weight', 'project_id', is_admin_state_up='admin_state_up')
    pool_id = resource.URI('pool_id')
    address = resource.Body('address')
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    protocol_port = resource.Body('protocol_port', type=int)
    subnet_id = resource.Body('subnet_id')
    weight = resource.Body('weight', type=int)