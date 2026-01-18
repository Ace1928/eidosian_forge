from neutron_lib.api.definitions import vpn
from neutron_lib.tests.unit.api.definitions import base
class VPNDefinitionTestCase(base.DefinitionBaseTestCase):
    extension_module = vpn
    extension_resources = ('vpnservices', 'ipsec_site_connections', 'ipsecpolicies', 'ikepolicies')
    extension_attributes = ('auth_algorithm', 'auth_mode', 'dpd', 'encapsulation_mode', 'encryption_algorithm', 'external_v4_ip', 'external_v6_ip', 'ike_version', 'ikepolicy_id', 'initiator', 'ipsecpolicy_id', 'lifetime', 'local_ep_group_id', 'local_id', 'mtu', 'peer_address', 'peer_cidrs', 'peer_ep_group_id', 'peer_id', 'pfs', 'phase1_negotiation_mode', 'psk', 'route_mode', 'router_id', 'subnet_id', 'transform_protocol', 'vpnservice_id')