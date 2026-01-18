from openstack import resource
class SfcPortPair(resource.Resource):
    resource_key = 'port_pair'
    resources_key = 'port_pairs'
    base_path = '/sfc/port_pairs'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'egress', 'ingress', 'project_id', 'tenant_id')
    description = resource.Body('description')
    name = resource.Body('name')
    ingress = resource.Body('ingress')
    egress = resource.Body('egress')
    service_function_parameters = resource.Body('service_function_parameters', type=dict)
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)