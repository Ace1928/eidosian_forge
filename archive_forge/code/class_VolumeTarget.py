from openstack.baremetal.v1 import _common
from openstack import resource
class VolumeTarget(_common.Resource):
    resources_key = 'targets'
    base_path = '/volume/targets'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters('node', 'detail', fields={'type': _common.fields_type})
    _max_microversion = '1.32'
    boot_index = resource.Body('boot_index')
    created_at = resource.Body('created_at')
    extra = resource.Body('extra')
    links = resource.Body('links', type=list)
    node_id = resource.Body('node_uuid')
    properties = resource.Body('properties')
    updated_at = resource.Body('updated_at')
    id = resource.Body('uuid', alternate_id=True)
    volume_id = resource.Body('volume_id')
    volume_type = resource.Body('volume_type')