from openstack.baremetal.v1 import _common
from openstack import resource
class VolumeConnector(_common.Resource):
    resources_key = 'connectors'
    base_path = '/volume/connectors'
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
    connector_id = resource.Body('connector_id')
    created_at = resource.Body('created_at')
    extra = resource.Body('extra')
    links = resource.Body('links', type=list)
    node_id = resource.Body('node_uuid')
    type = resource.Body('type')
    updated_at = resource.Body('updated_at')
    id = resource.Body('uuid', alternate_id=True)