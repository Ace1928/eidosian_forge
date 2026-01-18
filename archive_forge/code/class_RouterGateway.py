from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import subnet
from heat.engine import support
from heat.engine import translation
class RouterGateway(neutron.NeutronResource):
    support_status = support.SupportStatus(status=support.HIDDEN, message=_('Use the `external_gateway_info` property in the router resource to set up the gateway.'), version='5.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='2014.1'))
    PROPERTIES = ROUTER_ID, NETWORK_ID, NETWORK = ('router_id', 'network_id', 'network')
    properties_schema = {ROUTER_ID: properties.Schema(properties.Schema.STRING, _('ID of the router.'), required=True, constraints=[constraints.CustomConstraint('neutron.router')]), NETWORK_ID: properties.Schema(properties.Schema.STRING, support_status=support.SupportStatus(status=support.HIDDEN, message=_('Use property %s.') % NETWORK, version='9.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, version='2014.2')), constraints=[constraints.CustomConstraint('neutron.network')]), NETWORK: properties.Schema(properties.Schema.STRING, _('external network for the gateway.'), constraints=[constraints.CustomConstraint('neutron.network')])}

    def translation_rules(self, props):
        client_plugin = self.client_plugin()
        return [translation.TranslationRule(props, translation.TranslationRule.REPLACE, [self.NETWORK], value_path=[self.NETWORK_ID]), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.NETWORK], client_plugin=client_plugin, finder='find_resourceid_by_name_or_id', entity=client_plugin.RES_TYPE_NETWORK)]

    def add_dependencies(self, deps):
        super(RouterGateway, self).add_dependencies(deps)
        for resource in self.stack.values():
            if resource.has_interface('OS::Neutron::RouterInterface'):
                try:
                    dep_router_id = resource.properties[RouterInterface.ROUTER]
                    router_id = self.properties[self.ROUTER_ID]
                except (ValueError, TypeError):
                    continue
                if dep_router_id == router_id:
                    deps += (self, resource)
            if resource.has_interface('OS::Neutron::Subnet'):
                try:
                    dep_network = resource.properties[subnet.Subnet.NETWORK]
                    network = self.properties[self.NETWORK]
                except (ValueError, TypeError):
                    continue
                if dep_network == network:
                    deps += (self, resource)

    def handle_create(self):
        router_id = self.properties[self.ROUTER_ID]
        network_id = dict(self.properties).get(self.NETWORK)
        self.client().add_gateway_router(router_id, {'network_id': network_id})
        self.resource_id_set('%s:%s' % (router_id, network_id))

    def handle_delete(self):
        if not self.resource_id:
            return
        router_id, network_id = self.resource_id.split(':')
        with self.client_plugin().ignore_not_found:
            self.client().remove_gateway_router(router_id)