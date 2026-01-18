from openstack import resource
class SfcPortChain(resource.Resource):
    resource_key = 'port_chain'
    resources_key = 'port_chains'
    base_path = '/sfc/port_chains'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'tenant_id')
    description = resource.Body('description')
    name = resource.Body('name')
    port_pair_groups = resource.Body('port_pair_groups', type=list)
    flow_classifiers = resource.Body('flow_classifiers', type=list)
    chain_parameters = resource.Body('chain_parameters', type=dict)
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)