from openstack.baremetal.v1 import _common
from openstack import resource
class PortGroup(_common.Resource):
    resources_key = 'portgroups'
    base_path = '/portgroups'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters('node', 'address', fields={'type': _common.fields_type})
    _max_microversion = '1.26'
    address = resource.Body('address')
    created_at = resource.Body('created_at')
    extra = resource.Body('extra', type=dict)
    name = resource.Body('name')
    id = resource.Body('uuid', alternate_id=True)
    internal_info = resource.Body('internal_info')
    is_standalone_ports_supported = resource.Body('standalone_ports_supported', type=bool)
    links = resource.Body('links', type=list)
    mode = resource.Body('mode')
    node_id = resource.Body('node_uuid')
    ports = resource.Body('ports')
    properties = resource.Body('properties', type=dict)
    updated_at = resource.Body('updated_at')