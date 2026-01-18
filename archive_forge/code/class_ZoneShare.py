from openstack.dns.v2 import _base
from openstack import resource
class ZoneShare(_base.Resource):
    """DNS ZONE Share Resource"""
    resources_key = 'shared_zones'
    base_path = '/zones/%(zone_id)s/shares'
    allow_create = True
    allow_delete = True
    allow_fetch = True
    allow_list = True
    _query_mapping = resource.QueryParameters('target_project_id')
    zone_id = resource.URI('zone_id')
    created_at = resource.Body('created_at')
    updated_at = resource.Body('updated_at')
    project_id = resource.Body('project_id')
    target_project_id = resource.Body('target_project_id')