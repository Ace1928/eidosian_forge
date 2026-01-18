from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class NovaFloatingIpAssociation(resource.Resource):
    """A resource associates Nova floating IP with Nova server resource.

    Resource for associating existing Nova floating IP and Nova server.
    """
    deprecation_msg = _('Please use OS::Neutron::FloatingIPAssociation instead.')
    support_status = support.SupportStatus(status=support.HIDDEN, message=deprecation_msg, version='11.0.0', previous_status=support.SupportStatus(status=support.DEPRECATED, message=deprecation_msg, version='9.0.0', previous_status=support.SupportStatus(version='2014.1')))
    PROPERTIES = SERVER, FLOATING_IP = ('server_id', 'floating_ip')
    properties_schema = {SERVER: properties.Schema(properties.Schema.STRING, _('Server to assign floating IP to.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('nova.server')]), FLOATING_IP: properties.Schema(properties.Schema.STRING, _('ID of the floating IP to assign to the server.'), required=True, update_allowed=True)}
    default_client_name = 'nova'

    def get_reference_id(self):
        return self.physical_resource_name_or_FnGetRefId()

    def handle_create(self):
        self.client_plugin().associate_floatingip(self.properties[self.SERVER], self.properties[self.FLOATING_IP])
        self.resource_id_set(self.id)

    def handle_delete(self):
        if self.resource_id is None:
            return
        with self.client_plugin().ignore_not_found:
            self.client_plugin().dissociate_floatingip(self.properties[self.FLOATING_IP])

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            if self.FLOATING_IP in prop_diff:
                self.handle_delete()
            server_id = prop_diff.get(self.SERVER) or self.properties[self.SERVER]
            fl_ip_id = prop_diff.get(self.FLOATING_IP) or self.properties[self.FLOATING_IP]
            self.client_plugin().associate_floatingip(server_id, fl_ip_id)
            self.resource_id_set(self.id)