from openstack.common import tag
from openstack import resource
class LoadBalancerStats(resource.Resource):
    resource_key = 'stats'
    base_path = '/lbaas/loadbalancers/%(lb_id)s/stats'
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = False
    lb_id = resource.URI('lb_id')
    active_connections = resource.Body('active_connections', type=int)
    bytes_in = resource.Body('bytes_in', type=int)
    bytes_out = resource.Body('bytes_out', type=int)
    request_errors = resource.Body('request_errors', type=int)
    total_connections = resource.Body('total_connections', type=int)