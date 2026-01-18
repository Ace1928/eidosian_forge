from openstack import resource
class FirewallGroup(resource.Resource):
    resource_key = 'firewall_group'
    resources_key = 'firewall_groups'
    base_path = '/fwaas/firewall_groups'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('description', 'egress_firewall_policy_id', 'ingress_firewall_policy_id', 'name', 'shared', 'status', 'ports', 'project_id')
    admin_state_up = resource.Body('admin_state_up')
    description = resource.Body('description')
    egress_firewall_policy_id = resource.Body('egress_firewall_policy_id')
    ingress_firewall_policy_id = resource.Body('ingress_firewall_policy_id')
    id = resource.Body('id')
    name = resource.Body('name')
    ports = resource.Body('ports')
    project_id = resource.Body('project_id')
    shared = resource.Body('shared')
    status = resource.Body('status')