from openstack.common import tag
from openstack import resource
class L7Rule(resource.Resource, tag.TagMixin):
    resource_key = 'rule'
    resources_key = 'rules'
    base_path = '/lbaas/l7policies/%(l7policy_id)s/rules'
    allow_create = True
    allow_list = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    _query_mapping = resource.QueryParameters('compare_type', 'created_at', 'invert', 'key', 'project_id', 'provisioning_status', 'type', 'updated_at', 'rule_value', 'operating_status', is_admin_state_up='admin_state_up', l7_policy_id='l7policy_id', **tag.TagMixin._tag_query_parameters)
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    compare_type = resource.Body('compare_type')
    created_at = resource.Body('created_at')
    invert = resource.Body('invert', type=bool)
    key = resource.Body('key')
    l7_policy_id = resource.URI('l7policy_id')
    operating_status = resource.Body('operating_status')
    project_id = resource.Body('project_id')
    provisioning_status = resource.Body('provisioning_status')
    type = resource.Body('type')
    updated_at = resource.Body('updated_at')
    rule_value = resource.Body('value')