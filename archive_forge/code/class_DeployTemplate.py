from openstack.baremetal.v1 import _common
from openstack import resource
class DeployTemplate(_common.Resource):
    resources_key = 'deploy_templates'
    base_path = '/deploy_templates'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    allow_patch = True
    commit_method = 'PATCH'
    commit_jsonpatch = True
    _query_mapping = resource.QueryParameters('detail', fields={'type': _common.fields_type})
    _max_microversion = '1.55'
    name = resource.Body('name')
    created_at = resource.Body('created_at')
    extra = resource.Body('extra')
    links = resource.Body('links', type=list)
    steps = resource.Body('steps', type=list)
    updated_at = resource.Body('updated_at')
    id = resource.Body('uuid', alternate_id=True)