from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
class KeystoneUserRoleAssignment(resource.Resource, KeystoneRoleAssignmentMixin):
    """Resource for granting roles to a user.

    Resource for specifying users and their's roles.
    """
    support_status = support.SupportStatus(version='5.0.0', message=_('Supported versions: keystone v3'))
    default_client_name = 'keystone'
    PROPERTIES = USER, = ('user',)
    properties_schema = {USER: properties.Schema(properties.Schema.STRING, _('Name or id of keystone user.'), required=True, update_allowed=True, constraints=[constraints.CustomConstraint('keystone.user')])}
    properties_schema.update(KeystoneRoleAssignmentMixin.mixin_properties_schema)

    def client(self):
        return super(KeystoneUserRoleAssignment, self).client().client

    @property
    def user_id(self):
        try:
            return self.client_plugin().get_user_id(self.properties.get(self.USER))
        except Exception as ex:
            self.client_plugin().ignore_not_found(ex)
            return None

    def handle_create(self):
        self.create_assignment(user_id=self.user_id)

    def handle_update(self, json_snippet, tmpl_diff, prop_diff):
        self.update_assignment(user_id=self.user_id, prop_diff=prop_diff)

    def handle_delete(self):
        with self.client_plugin().ignore_not_found:
            self.delete_assignment(user_id=self.user_id)

    def validate(self):
        super(KeystoneUserRoleAssignment, self).validate()
        self.validate_assignment_properties()