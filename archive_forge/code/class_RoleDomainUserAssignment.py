from openstack import resource
class RoleDomainUserAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/domains/%(domain_id)s/users/%(user_id)s/roles'
    allow_list = True
    name = resource.Body('name')
    links = resource.Body('links')
    domain_id = resource.URI('domain_id')
    user_id = resource.URI('user_id')