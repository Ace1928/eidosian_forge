from openstack import resource
class VpnIkePolicy(resource.Resource):
    """VPN IKE policy extension."""
    resource_key = 'ikepolicy'
    resources_key = 'ikepolicies'
    base_path = '/vpn/ikepolicies'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('auth_algorithm', 'description', 'encryption_algorithm', 'ike_version', 'name', 'pfs', 'project_id', 'phase1_negotiation_mode')
    auth_algorithm = resource.Body('auth_algorithm')
    description = resource.Body('description')
    encryption_algorithm = resource.Body('encryption_algorithm')
    ike_version = resource.Body('ike_version')
    lifetime = resource.Body('lifetime', type=dict)
    name = resource.Body('name')
    pfs = resource.Body('pfs')
    project_id = resource.Body('project_id')
    phase1_negotiation_mode = resource.Body('phase1_negotiation_mode')
    units = resource.Body('units')
    value = resource.Body('value', type=int)