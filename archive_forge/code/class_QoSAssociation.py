from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
class QoSAssociation(resource.Resource):
    """A resource to associate cinder QoS specs with volume types.

    Usage of this resource restricted to admins only by default policy.
    """
    support_status = support.SupportStatus(version='8.0.0')
    default_client_name = 'cinder'
    required_service_extension = 'qos-specs'
    PROPERTIES = QOS_SPECS, VOLUME_TYPES = ('qos_specs', 'volume_types')
    properties_schema = {QOS_SPECS: properties.Schema(properties.Schema.STRING, _('ID or Name of the QoS specs.'), required=True, constraints=[constraints.CustomConstraint('cinder.qos_specs')]), VOLUME_TYPES: properties.Schema(properties.Schema.LIST, _('List of volume type IDs or Names to be attached to QoS specs.'), schema=properties.Schema(properties.Schema.STRING, _('A volume type to attach specs.'), constraints=[constraints.CustomConstraint('cinder.vtype')]), update_allowed=True, required=True)}

    def translation_rules(self, props):
        return [translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.VOLUME_TYPES], client_plugin=self.client_plugin(), finder='get_volume_type'), translation.TranslationRule(props, translation.TranslationRule.RESOLVE, [self.QOS_SPECS], client_plugin=self.client_plugin(), finder='get_qos_specs')]

    def _find_diff(self, update_prps, stored_prps):
        add_prps = list(set(update_prps or []) - set(stored_prps or []))
        remove_prps = list(set(stored_prps or []) - set(update_prps or []))
        return (add_prps, remove_prps)

    def handle_create(self):
        for vt in self.properties[self.VOLUME_TYPES]:
            self.client().qos_specs.associate(self.properties[self.QOS_SPECS], vt)
        self.resource_id_set(self.uuid)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        """Associate volume types to QoS."""
        qos_specs = self.properties[self.QOS_SPECS]
        new_associate_vts = prop_diff.get(self.VOLUME_TYPES)
        old_associate_vts = self.properties[self.VOLUME_TYPES]
        add_associate_vts, remove_associate_vts = self._find_diff(new_associate_vts, old_associate_vts)
        for vt in add_associate_vts:
            self.client().qos_specs.associate(qos_specs, vt)
        for vt in remove_associate_vts:
            self.client().qos_specs.disassociate(qos_specs, vt)

    def handle_delete(self):
        volume_types = self.properties[self.VOLUME_TYPES]
        for vt in volume_types:
            self.client().qos_specs.disassociate(self.properties[self.QOS_SPECS], vt)