from openstack.baremetal.v1 import _common
from openstack import resource
class Chassis(_common.Resource):
    resources_key = 'chassis'
    base_path = '/chassis'
    _max_microversion = '1.8'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters(fields={'type': _common.fields_type})
    created_at = resource.Body('created_at')
    description = resource.Body('description')
    extra = resource.Body('extra')
    id = resource.Body('uuid', alternate_id=True)
    links = resource.Body('links', type=list)
    nodes = resource.Body('nodes', type=list)
    updated_at = resource.Body('updated_at')