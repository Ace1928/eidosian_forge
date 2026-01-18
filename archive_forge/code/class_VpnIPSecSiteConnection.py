from openstack import resource
class VpnIPSecSiteConnection(resource.Resource):
    resource_key = 'ipsec_site_connection'
    resources_key = 'ipsec_site_connections'
    base_path = '/vpn/ipsec-site-connections'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('auth_mode', 'description', 'ikepolicy_id', 'ipsecpolicy_id', 'initiator', 'local_ep_group_id', 'peer_address', 'local_id', 'mtu', 'name', 'peer_id', 'project_id', 'psk', 'peer_ep_group_id', 'route_mode', 'vpnservice_id', 'status', is_admin_state_up='admin_state_up')
    action = resource.Body('action')
    auth_mode = resource.Body('auth_mode')
    description = resource.Body('description')
    dpd = resource.Body('dpd', type=dict)
    is_admin_state_up = resource.Body('admin_state_up', type=bool)
    ikepolicy_id = resource.Body('ikepolicy_id')
    initiator = resource.Body('initiator')
    ipsecpolicy_id = resource.Body('ipsecpolicy_id')
    interval = resource.Body('interval', type=int)
    local_ep_group_id = resource.Body('local_ep_group_id')
    peer_address = resource.Body('peer_address')
    local_id = resource.Body('local_id')
    mtu = resource.Body('mtu', type=int)
    name = resource.Body('name')
    peer_id = resource.Body('peer_id')
    peer_cidrs = resource.Body('peer_cidrs', type=list)
    project_id = resource.Body('tenant_id')
    psk = resource.Body('psk')
    peer_ep_group_id = resource.Body('peer_ep_group_id')
    route_mode = resource.Body('route_mode')
    status = resource.Body('status')
    timeout = resource.Body('timeout', type=int)
    vpnservice_id = resource.Body('vpnservice_id')