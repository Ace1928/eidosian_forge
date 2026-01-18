from openstack import resource
class RBACPolicy(resource.Resource):
    resource_key = 'rbac_policy'
    resources_key = 'rbac_policies'
    base_path = '/rbac-policies'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('action', 'object_id', 'object_type', 'project_id', 'target_project_id', target_project_id='target_tenant')
    object_id = resource.Body('object_id')
    target_project_id = resource.Body('target_tenant')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    object_type = resource.Body('object_type')
    action = resource.Body('action')