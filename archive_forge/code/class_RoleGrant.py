from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RoleGrant(_messages.Message):
    """This configuration defines all the Cloud IAM roles that needs to be
  granted to a particular Google Cloud resource for the selected principal
  like service account. These configurations will let UI display to customers
  what IAM roles need to be granted by them. Or these configurations can be
  used by the UI to render a 'grant' button to do the same on behalf of the
  user.

  Enums:
    PrincipalValueValuesEnum: Prinicipal/Identity for whom the role need to
      assigned.

  Fields:
    helperTextTemplate: Template that UI can use to provide helper text to
      customers.
    principal: Prinicipal/Identity for whom the role need to assigned.
    resource: Resource on which the roles needs to be granted for the
      principal.
    roles: List of roles that need to be granted.
  """

    class PrincipalValueValuesEnum(_messages.Enum):
        """Prinicipal/Identity for whom the role need to assigned.

    Values:
      PRINCIPAL_UNSPECIFIED: Value type is not specified.
      CONNECTOR_SA: Service Account used for Connector workload identity This
        is either the default service account if unspecified or Service
        Account provided by Customers through BYOSA.
    """
        PRINCIPAL_UNSPECIFIED = 0
        CONNECTOR_SA = 1
    helperTextTemplate = _messages.StringField(1)
    principal = _messages.EnumField('PrincipalValueValuesEnum', 2)
    resource = _messages.MessageField('Resource', 3)
    roles = _messages.StringField(4, repeated=True)