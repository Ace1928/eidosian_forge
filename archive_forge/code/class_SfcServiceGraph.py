from openstack import resource
class SfcServiceGraph(resource.Resource):
    resource_key = 'service_graph'
    resources_key = 'service_graphs'
    base_path = '/sfc/service_graphs'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'tenant_id')
    description = resource.Body('description')
    name = resource.Body('name')
    port_chains = resource.Body('port_chains')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)