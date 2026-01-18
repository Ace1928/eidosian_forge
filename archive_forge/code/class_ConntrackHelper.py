from openstack import resource
class ConntrackHelper(resource.Resource):
    resource_key = 'conntrack_helper'
    resources_key = 'conntrack_helpers'
    base_path = '/routers/%(router_id)s/conntrack_helpers'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    router_id = resource.URI('router_id')
    helper = resource.Body('helper')
    protocol = resource.Body('protocol')
    port = resource.Body('port')