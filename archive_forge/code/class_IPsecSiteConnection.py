from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
class IPsecSiteConnection(neutron.NeutronResource):
    """A resource for IPsec site connection in Neutron.

    This resource has details for the site-to-site IPsec connection, including
    the peer CIDRs, MTU, peer address, DPD settings and status.
    """
    required_service_extension = 'vpnaas'
    entity = 'ipsec_site_connection'
    PROPERTIES = NAME, DESCRIPTION, PEER_ADDRESS, PEER_ID, PEER_CIDRS, MTU, DPD, PSK, INITIATOR, ADMIN_STATE_UP, IKEPOLICY_ID, IPSECPOLICY_ID, VPNSERVICE_ID = ('name', 'description', 'peer_address', 'peer_id', 'peer_cidrs', 'mtu', 'dpd', 'psk', 'initiator', 'admin_state_up', 'ikepolicy_id', 'ipsecpolicy_id', 'vpnservice_id')
    _DPD_KEYS = DPD_ACTIONS, DPD_INTERVAL, DPD_TIMEOUT = ('actions', 'interval', 'timeout')
    ATTRIBUTES = ADMIN_STATE_UP_ATTR, AUTH_MODE, DESCRIPTION_ATTR, DPD_ATTR, IKEPOLICY_ID_ATTR, INITIATOR_ATTR, IPSECPOLICY_ID_ATTR, MTU_ATTR, NAME_ATTR, PEER_ADDRESS_ATTR, PEER_CIDRS_ATTR, PEER_ID_ATTR, PSK_ATTR, ROUTE_MODE, STATUS, TENANT_ID, VPNSERVICE_ID_ATTR = ('admin_state_up', 'auth_mode', 'description', 'dpd', 'ikepolicy_id', 'initiator', 'ipsecpolicy_id', 'mtu', 'name', 'peer_address', 'peer_cidrs', 'peer_id', 'psk', 'route_mode', 'status', 'tenant_id', 'vpnservice_id')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name for the ipsec site connection.'), update_allowed=True), DESCRIPTION: properties.Schema(properties.Schema.STRING, _('Description for the ipsec site connection.'), update_allowed=True), PEER_ADDRESS: properties.Schema(properties.Schema.STRING, _('Remote branch router public IPv4 address or IPv6 address or FQDN.'), required=True), PEER_ID: properties.Schema(properties.Schema.STRING, _('Remote branch router identity.'), required=True), PEER_CIDRS: properties.Schema(properties.Schema.LIST, _('Remote subnet(s) in CIDR format.'), required=True, schema=properties.Schema(properties.Schema.STRING, constraints=[constraints.CustomConstraint('net_cidr')])), MTU: properties.Schema(properties.Schema.INTEGER, _('Maximum transmission unit size (in bytes) for the ipsec site connection.'), default=1500), DPD: properties.Schema(properties.Schema.MAP, _('Dead Peer Detection protocol configuration for the ipsec site connection.'), schema={DPD_ACTIONS: properties.Schema(properties.Schema.STRING, _('Controls DPD protocol mode.'), default='hold', constraints=[constraints.AllowedValues(['clear', 'disabled', 'hold', 'restart', 'restart-by-peer'])]), DPD_INTERVAL: properties.Schema(properties.Schema.INTEGER, _('Number of seconds for the DPD delay.'), default=30), DPD_TIMEOUT: properties.Schema(properties.Schema.INTEGER, _('Number of seconds for the DPD timeout.'), default=120)}), PSK: properties.Schema(properties.Schema.STRING, _('Pre-shared key string for the ipsec site connection.'), required=True), INITIATOR: properties.Schema(properties.Schema.STRING, _('Initiator state in lowercase for the ipsec site connection.'), default='bi-directional', constraints=[constraints.AllowedValues(['bi-directional', 'response-only'])]), ADMIN_STATE_UP: properties.Schema(properties.Schema.BOOLEAN, _('Administrative state for the ipsec site connection.'), default=True, update_allowed=True), IKEPOLICY_ID: properties.Schema(properties.Schema.STRING, _('Unique identifier for the ike policy associated with the ipsec site connection.'), required=True), IPSECPOLICY_ID: properties.Schema(properties.Schema.STRING, _('Unique identifier for the ipsec policy associated with the ipsec site connection.'), required=True), VPNSERVICE_ID: properties.Schema(properties.Schema.STRING, _('Unique identifier for the vpn service associated with the ipsec site connection.'), required=True)}
    attributes_schema = {ADMIN_STATE_UP_ATTR: attributes.Schema(_('The administrative state of the ipsec site connection.'), type=attributes.Schema.STRING), AUTH_MODE: attributes.Schema(_('The authentication mode of the ipsec site connection.'), type=attributes.Schema.STRING), DESCRIPTION_ATTR: attributes.Schema(_('The description of the ipsec site connection.'), type=attributes.Schema.STRING), DPD_ATTR: attributes.Schema(_('The dead peer detection protocol configuration of the ipsec site connection.'), type=attributes.Schema.MAP), IKEPOLICY_ID_ATTR: attributes.Schema(_('The unique identifier of ike policy associated with the ipsec site connection.'), type=attributes.Schema.STRING), INITIATOR_ATTR: attributes.Schema(_('The initiator of the ipsec site connection.'), type=attributes.Schema.STRING), IPSECPOLICY_ID_ATTR: attributes.Schema(_('The unique identifier of ipsec policy associated with the ipsec site connection.'), type=attributes.Schema.STRING), MTU_ATTR: attributes.Schema(_('The maximum transmission unit size (in bytes) of the ipsec site connection.'), type=attributes.Schema.STRING), NAME_ATTR: attributes.Schema(_('The name of the ipsec site connection.'), type=attributes.Schema.STRING), PEER_ADDRESS_ATTR: attributes.Schema(_('The remote branch router public IPv4 address or IPv6 address or FQDN.'), type=attributes.Schema.STRING), PEER_CIDRS_ATTR: attributes.Schema(_('The remote subnet(s) in CIDR format of the ipsec site connection.'), type=attributes.Schema.LIST), PEER_ID_ATTR: attributes.Schema(_('The remote branch router identity of the ipsec site connection.'), type=attributes.Schema.STRING), PSK_ATTR: attributes.Schema(_('The pre-shared key string of the ipsec site connection.'), type=attributes.Schema.STRING), ROUTE_MODE: attributes.Schema(_('The route mode of the ipsec site connection.'), type=attributes.Schema.STRING), STATUS: attributes.Schema(_('The status of the ipsec site connection.'), type=attributes.Schema.STRING), TENANT_ID: attributes.Schema(_('The unique identifier of the tenant owning the ipsec site connection.'), type=attributes.Schema.STRING), VPNSERVICE_ID_ATTR: attributes.Schema(_('The unique identifier of vpn service associated with the ipsec site connection.'), type=attributes.Schema.STRING)}

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        ipsec_site_connection = self.client().create_ipsec_site_connection({'ipsec_site_connection': props})['ipsec_site_connection']
        self.resource_id_set(ipsec_site_connection['id'])

    def check_create_complete(self, data):
        attributes = self._show_resource()
        status = attributes['status']
        if status == 'PENDING_CREATE':
            return False
        elif status == 'ACTIVE':
            return True
        elif status == 'ERROR':
            raise exception.ResourceInError(resource_status=status, status_reason=_('Error in IPsecSiteConnection'))
        else:
            raise exception.ResourceUnknownStatus(resource_status=status, result=_('IPsecSiteConnection creation failed'))

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            self.client().update_ipsec_site_connection(self.resource_id, {'ipsec_site_connection': prop_diff})

    def handle_delete(self):
        try:
            self.client().delete_ipsec_site_connection(self.resource_id)
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
        else:
            return True