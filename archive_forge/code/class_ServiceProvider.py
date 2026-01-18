from openstack import resource
class ServiceProvider(resource.Resource):
    resources_key = 'service_providers'
    base_path = '/service-providers'
    _allow_unknown_attrs_in_body = True
    allow_create = False
    allow_fetch = False
    allow_commit = False
    allow_delete = False
    allow_list = True
    _query_mapping = resource.QueryParameters('service_type', 'name', is_default='default')
    service_type = resource.Body('service_type')
    name = resource.Body('name')
    is_default = resource.Body('default', type=bool)