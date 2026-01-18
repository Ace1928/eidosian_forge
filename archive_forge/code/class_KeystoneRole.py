from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import translation
class KeystoneRole(resource.Resource):
    """Heat Template Resource for Keystone Role.

    Roles dictate the level of authorization the end user can obtain. Roles can
    be granted at either the domain or project level. Role can be assigned to
    the individual user or at the group level. Role name is unique within the
    owning domain.
    """
    support_status = support.SupportStatus(version='2015.1', message=_('Supported versions: keystone v3'))
    default_client_name = 'keystone'
    entity = 'roles'
    PROPERTIES = NAME, DOMAIN = ('name', 'domain')
    properties_schema = {NAME: properties.Schema(properties.Schema.STRING, _('Name of keystone role.'), update_allowed=True), DOMAIN: properties.Schema(properties.Schema.STRING, _('Name or id of keystone domain.'), constraints=[constraints.CustomConstraint('keystone.domain')], support_status=support.SupportStatus(version='16.0.0'))}

    def translation_rules(self, properties):
        return [translation.TranslationRule(properties, translation.TranslationRule.RESOLVE, [self.DOMAIN], client_plugin=self.client_plugin(), finder='get_domain_id')]

    def client(self):
        return super(KeystoneRole, self).client().client

    def handle_create(self):
        role_name = self.properties[self.NAME] or self.physical_resource_name()
        domain = self.properties[self.DOMAIN]
        role = self.client().roles.create(name=role_name, domain=domain)
        self.resource_id_set(role.id)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        if prop_diff:
            if self.NAME in prop_diff:
                name = prop_diff[self.NAME] or self.physical_resource_name()
                self.client().roles.update(role=self.resource_id, name=name)