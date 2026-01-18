from openstack import resource
class MeteringLabel(resource.Resource):
    resource_key = 'metering_label'
    resources_key = 'metering_labels'
    base_path = '/metering/metering-labels'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'name', 'project_id', 'sort_key', 'sort_dir', is_shared='shared')
    description = resource.Body('description')
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    is_shared = resource.Body('shared', type=bool)