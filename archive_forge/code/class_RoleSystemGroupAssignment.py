from openstack import resource
class RoleSystemGroupAssignment(resource.Resource):
    resource_key = 'role'
    resources_key = 'roles'
    base_path = '/system/groups/%(group_id)s/roles'
    allow_list = True
    group_id = resource.URI('group_id')
    system_id = resource.URI('system_id')