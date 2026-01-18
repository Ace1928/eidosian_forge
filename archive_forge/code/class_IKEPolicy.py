from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
class IKEPolicy(neutron.NeutronResource):
    """A resource for IKE policy in Neutron.

    The Internet Key Exchange policy identifies the authentication and
    encryption algorithm used during phase one and phase two negotiation of a
    VPN connection.
    """
    required_service_extension = 'vpnaas'
    entity = 'ikepolicy'
    PROPERTIES = NAME, DESCRIPTION, AUTH_ALGORITHM, ENCRYPTION_ALGORITHM, PHASE1_NEGOTIATION_MODE, LIFETIME, PFS, IKE_VERSION = ('name', 'description', 'auth_algorithm', 'encryption_algorithm', 'phase1_negotiation_mode', 'lifetime', 'pfs', 'ike_version')
    _LIFETIME_KEYS = LIFETIME_UNITS, LIFETIME_VALUE = ('units', 'value')
    ATTRIBUTES = AUTH_ALGORITHM_ATTR, DESCRIPTION_ATTR, ENCRYPTION_ALGORITHM_ATTR, IKE_VERSION_ATTR, LIFETIME_ATTR, NAME_ATTR, PFS_ATTR, PHASE1_NEGOTIATION_MODE_ATTR, TENANT_ID = ('auth_algorithm', 'description', 'encryption_algorithm', 'ike_version', 'lifetime', 'name', 'pfs', 'phase1_negotiation_mode', 'tenant_id')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name for the ike policy.'), update_allowed=True), DESCRIPTION: properties.Schema(properties.Schema.STRING, _('Description for the ike policy.'), update_allowed=True), AUTH_ALGORITHM: properties.Schema(properties.Schema.STRING, _('Authentication hash algorithm for the ike policy.'), default='sha1', constraints=[constraints.AllowedValues(['sha1', 'sha256', 'sha384', 'sha512'])], update_allowed=True), ENCRYPTION_ALGORITHM: properties.Schema(properties.Schema.STRING, _('Encryption algorithm for the ike policy.'), default='aes-128', constraints=[constraints.AllowedValues(['3des', 'aes-128', 'aes-192', 'aes-256'])], update_allowed=True), PHASE1_NEGOTIATION_MODE: properties.Schema(properties.Schema.STRING, _('Negotiation mode for the ike policy.'), default='main', constraints=[constraints.AllowedValues(['main'])]), LIFETIME: properties.Schema(properties.Schema.MAP, _('Safety assessment lifetime configuration for the ike policy.'), update_allowed=True, schema={LIFETIME_UNITS: properties.Schema(properties.Schema.STRING, _('Safety assessment lifetime units.'), default='seconds', constraints=[constraints.AllowedValues(['seconds', 'kilobytes'])]), LIFETIME_VALUE: properties.Schema(properties.Schema.INTEGER, _('Safety assessment lifetime value in specified units.'), default=3600)}), PFS: properties.Schema(properties.Schema.STRING, _('Perfect forward secrecy in lowercase for the ike policy.'), default='group5', constraints=[constraints.AllowedValues(['group2', 'group5', 'group14'])], update_allowed=True), IKE_VERSION: properties.Schema(properties.Schema.STRING, _('Version for the ike policy.'), default='v1', constraints=[constraints.AllowedValues(['v1', 'v2'])], update_allowed=True)}
    attributes_schema = {AUTH_ALGORITHM_ATTR: attributes.Schema(_('The authentication hash algorithm used by the ike policy.'), type=attributes.Schema.STRING), DESCRIPTION_ATTR: attributes.Schema(_('The description of the ike policy.'), type=attributes.Schema.STRING), ENCRYPTION_ALGORITHM_ATTR: attributes.Schema(_('The encryption algorithm used by the ike policy.'), type=attributes.Schema.STRING), IKE_VERSION_ATTR: attributes.Schema(_('The version of the ike policy.'), type=attributes.Schema.STRING), LIFETIME_ATTR: attributes.Schema(_('The safety assessment lifetime configuration for the ike policy.'), type=attributes.Schema.MAP), NAME_ATTR: attributes.Schema(_('The name of the ike policy.'), type=attributes.Schema.STRING), PFS_ATTR: attributes.Schema(_('The perfect forward secrecy of the ike policy.'), type=attributes.Schema.STRING), PHASE1_NEGOTIATION_MODE_ATTR: attributes.Schema(_('The negotiation mode of the ike policy.'), type=attributes.Schema.STRING), TENANT_ID: attributes.Schema(_('The unique identifier of the tenant owning the ike policy.'), type=attributes.Schema.STRING)}

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        ikepolicy = self.client().create_ikepolicy({'ikepolicy': props})['ikepolicy']
        self.resource_id_set(ikepolicy['id'])

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            self.client().update_ikepolicy(self.resource_id, {'ikepolicy': prop_diff})

    def handle_delete(self):
        try:
            self.client().delete_ikepolicy(self.resource_id)
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
        else:
            return True