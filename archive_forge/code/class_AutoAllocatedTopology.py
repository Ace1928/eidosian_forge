from openstack import resource
class AutoAllocatedTopology(resource.Resource):
    resource_name = 'auto_allocated_topology'
    resource_key = 'auto_allocated_topology'
    base_path = '/auto-allocated-topology'
    _allow_unknown_attrs_in_body = True
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = False
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)