from openstack.dns.v2 import _base
from openstack import resource
class ZoneTransferBase(_base.Resource):
    """DNS Zone Transfer Request/Accept Base Resource"""
    _query_mapping = resource.QueryParameters('status')
    created_at = resource.Body('created_at')
    key = resource.Body('key')
    project_id = resource.Body('project_id')
    status = resource.Body('status')
    updated_at = resource.Body('updated_at')
    version = resource.Body('version', type=int)
    zone_id = resource.Body('zone_id')