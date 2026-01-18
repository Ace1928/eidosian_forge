from openstack import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients.os import openstacksdk as sdk_plugin
from heat.engine import constraints
class PolicyTypeConstraint(constraints.BaseCustomConstraint):
    expected_exceptions = (exception.StackValidationFailed,)

    def validate_with_client(self, client, value):
        conn = client.client(CLIENT_NAME)
        type_list = conn.policy_types()
        names = [pt.name for pt in type_list]
        if value not in names:
            not_found_message = _("Unable to find senlin policy type '%(pt)s', available policy types are %(pts)s.") % {'pt': value, 'pts': names}
            raise exception.StackValidationFailed(message=not_found_message)