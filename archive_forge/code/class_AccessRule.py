from openstack import resource
class AccessRule(resource.Resource):
    resource_key = 'access_rule'
    resources_key = 'access_rules'
    base_path = '/users/%(user_id)s/access_rules'
    allow_fetch = True
    allow_delete = True
    allow_list = True
    links = resource.Body('links')
    method = resource.Body('method')
    path = resource.Body('path')
    service = resource.Body('service')
    user_id = resource.URI('user_id')