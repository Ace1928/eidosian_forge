from openstack import resource
class ServiceProfile(resource.Resource):
    resource_key = 'service_profile'
    resources_key = 'service_profiles'
    base_path = '/service_profiles'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'driver', 'project_id', is_enabled='enabled')
    description = resource.Body('description')
    driver = resource.Body('driver')
    is_enabled = resource.Body('enabled', type=bool)
    meta_info = resource.Body('metainfo')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)