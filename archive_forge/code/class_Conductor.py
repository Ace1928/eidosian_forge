from openstack.baremetal.v1 import _common
from openstack import resource
class Conductor(_common.Resource):
    resources_key = 'conductors'
    base_path = '/conductors'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True
    allow_patch = False
    _query_mapping = resource.QueryParameters('detail', fields={'type': _common.fields_type})
    _max_microversion = '1.49'
    created_at = resource.Body('created_at')
    updated_at = resource.Body('updated_at')
    hostname = resource.Body('hostname')
    conductor_group = resource.Body('conductor_group')
    alive = resource.Body('alive', type=bool)
    links = resource.Body('links', type=list)
    drivers = resource.Body('drivers', type=list)