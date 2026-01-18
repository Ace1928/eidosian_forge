from openstack.common import tag
from openstack import resource
class L7Policy(resource.Resource, tag.TagMixin):
    resource_key = 'l7policy'
    resources_key = 'l7policies'
    base_path = '/lbaas/l7policies'
    allow_create = True
    allow_list = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    _query_mapping = resource.QueryParameters('action', 'description', 'listener_id', 'name', 'position', 'redirect_pool_id', 'redirect_url', 'provisioning_status', 'operating_status', 'redirect_prefix', 'project_id', is_admin_state_up='admin_state_up', **tag.TagMixin._tag_query_parameters)
    action = resource.Body('action')
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    listener_id = resource.Body('listener_id')
    name = resource.Body('name')
    operating_status = resource.Body('operating_status')
    position = resource.Body('position', type=int)
    project_id = resource.Body('project_id')
    provisioning_status = resource.Body('provisioning_status')
    redirect_pool_id = resource.Body('redirect_pool_id')
    redirect_prefix = resource.Body('redirect_prefix')
    redirect_url = resource.Body('redirect_url')
    rules = resource.Body('rules', type=list)
    updated_at = resource.Body('updated_at')