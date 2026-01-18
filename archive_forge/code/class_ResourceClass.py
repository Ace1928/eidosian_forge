from openstack import resource
class ResourceClass(resource.Resource):
    resource_key = None
    resources_key = 'resource_classes'
    base_path = '/resource_classes'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _max_microversion = '1.2'
    name = resource.Body('name', alternate_id=True)