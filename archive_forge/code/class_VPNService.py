from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
class VPNService(neutron.NeutronResource):
    """A resource for VPN service in Neutron.

    VPN service is a high level object that associates VPN with a specific
    subnet and router.
    """
    required_service_extension = 'vpnaas'
    entity = 'vpnservice'
    PROPERTIES = NAME, DESCRIPTION, ADMIN_STATE_UP, SUBNET_ID, SUBNET, ROUTER_ID, ROUTER = ('name', 'description', 'admin_state_up', 'subnet_id', 'subnet', 'router_id', 'router')
    ATTRIBUTES = ADMIN_STATE_UP_ATTR, DESCRIPTION_ATTR, NAME_ATTR, ROUTER_ID_ATTR, STATUS, SUBNET_ID_ATTR, TENANT_ID = ('admin_state_up', 'description', 'name', 'router_id', 'status', 'subnet_id', 'tenant_id')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name for the vpn service.'), update_allowed=True), DESCRIPTION: properties.Schema(properties.Schema.STRING, _('Description for the vpn service.'), update_allowed=True), ADMIN_STATE_UP: properties.Schema(properties.Schema.BOOLEAN, _('Administrative state for the vpn service.'), default=True, update_allowed=True), SUBNET_ID: properties.Schema(properties.Schema.STRING, support_status=support.SupportStatus(status=support.HIDDEN, message=_('Use property %s.') % SUBNET, version='5.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='2014.2')), constraints=[constraints.CustomConstraint('neutron.subnet')]), SUBNET: properties.Schema(properties.Schema.STRING, _('Subnet in which the vpn service will be created.'), support_status=support.SupportStatus(version='2014.2'), required=True, constraints=[constraints.CustomConstraint('neutron.subnet')]), ROUTER_ID: properties.Schema(properties.Schema.STRING, _('Unique identifier for the router to which the vpn service will be inserted.'), support_status=support.SupportStatus(status=support.HIDDEN, version='6.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, message=_('Use property %s') % ROUTER, version='2015.1', previous_status=support.SupportStatus(version='2013.2'))), constraints=[constraints.CustomConstraint('neutron.router')]), ROUTER: properties.Schema(properties.Schema.STRING, _('The router to which the vpn service will be inserted.'), support_status=support.SupportStatus(version='2015.1'), required=True, constraints=[constraints.CustomConstraint('neutron.router')])}
    attributes_schema = {ADMIN_STATE_UP_ATTR: attributes.Schema(_('The administrative state of the vpn service.'), type=attributes.Schema.STRING), DESCRIPTION_ATTR: attributes.Schema(_('The description of the vpn service.'), type=attributes.Schema.STRING), NAME_ATTR: attributes.Schema(_('The name of the vpn service.'), type=attributes.Schema.STRING), ROUTER_ID_ATTR: attributes.Schema(_('The unique identifier of the router to which the vpn service was inserted.'), type=attributes.Schema.STRING), STATUS: attributes.Schema(_('The status of the vpn service.'), type=attributes.Schema.STRING), SUBNET_ID_ATTR: attributes.Schema(_('The unique identifier of the subnet in which the vpn service was created.'), type=attributes.Schema.STRING), TENANT_ID: attributes.Schema(_('The unique identifier of the tenant owning the vpn service.'), type=attributes.Schema.STRING)}

    def translation_rules(self, props):
        client_plugin = self.client_plugin()
        return [translation.TranslationRule(props, translation.TranslationRule.REPLACE, [self.SUBNET], value_path=[self.SUBNET_ID]), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.SUBNET], client_plugin=client_plugin, finder='find_resourceid_by_name_or_id', entity=client_plugin.RES_TYPE_SUBNET), translation.TranslationRule(props, translation.TranslationRule.REPLACE, [self.ROUTER], value_path=[self.ROUTER_ID]), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.ROUTER], client_plugin=client_plugin, finder='find_resourceid_by_name_or_id', entity=client_plugin.RES_TYPE_ROUTER)]

    def handle_create(self):
        props = self.prepare_properties(self.properties, self.physical_resource_name())
        props['subnet_id'] = props.pop(self.SUBNET)
        props['router_id'] = props.pop(self.ROUTER)
        vpnservice = self.client().create_vpnservice({'vpnservice': props})['vpnservice']
        self.resource_id_set(vpnservice['id'])

    def check_create_complete(self, data):
        attributes = self._show_resource()
        status = attributes['status']
        if status == 'PENDING_CREATE':
            return False
        elif status == 'ACTIVE':
            return True
        elif status == 'ERROR':
            raise exception.ResourceInError(resource_status=status, status_reason=_('Error in VPNService'))
        else:
            raise exception.ResourceUnknownStatus(resource_status=status, result=_('VPNService creation failed'))

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            self.prepare_update_properties(prop_diff)
            self.client().update_vpnservice(self.resource_id, {'vpnservice': prop_diff})

    def handle_delete(self):
        try:
            self.client().delete_vpnservice(self.resource_id)
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
        else:
            return True