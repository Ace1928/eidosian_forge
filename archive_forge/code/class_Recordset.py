from openstack.dns.v2 import _base
from openstack import resource
class Recordset(_base.Resource):
    """DNS Recordset Resource"""
    resources_key = 'recordsets'
    base_path = '/zones/%(zone_id)s/recordsets'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'type', 'ttl', 'data', 'status', 'description', 'limit', 'marker')
    action = resource.Body('action')
    created_at = resource.Body('create_at')
    description = resource.Body('description')
    links = resource.Body('links', type=dict)
    name = resource.Body('name')
    project_id = resource.Body('project_id')
    records = resource.Body('records', type=list)
    status = resource.Body('status')
    ttl = resource.Body('ttl', type=int)
    type = resource.Body('type')
    updated_at = resource.Body('updated_at')
    zone_id = resource.URI('zone_id')
    zone_name = resource.Body('zone_name')