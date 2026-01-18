from openstack.common import tag
from openstack import resource
class SubnetPool(resource.Resource, tag.TagMixin):
    resource_key = 'subnetpool'
    resources_key = 'subnetpools'
    base_path = '/subnetpools'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('address_scope_id', 'description', 'ip_version', 'is_default', 'name', 'project_id', 'sort_key', 'sort_dir', is_shared='shared', **tag.TagMixin._tag_query_parameters)
    address_scope_id = resource.Body('address_scope_id')
    created_at = resource.Body('created_at')
    default_prefix_length = resource.Body('default_prefixlen', type=int)
    default_quota = resource.Body('default_quota', type=int)
    description = resource.Body('description')
    ip_version = resource.Body('ip_version', type=int)
    is_default = resource.Body('is_default', type=bool)
    is_shared = resource.Body('shared', type=bool)
    maximum_prefix_length = resource.Body('max_prefixlen', type=int)
    minimum_prefix_length = resource.Body('min_prefixlen', type=int)
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    prefixes = resource.Body('prefixes', type=list)
    revision_number = resource.Body('revision_number', type=int)
    updated_at = resource.Body('updated_at')