from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfigIngressFrom(_messages.Message):
    """Defines the conditions under which an IngressPolicy matches a request.
  Conditions are based on information about the source of the request. The
  request must satisfy what is defined in `sources` AND identity related
  fields in order to match.

  Enums:
    IdentityTypeValueValuesEnum: Specifies the type of identities that are
      allowed access from outside the perimeter. If left unspecified, then
      members of `identities` field will be allowed access.

  Fields:
    identities: A list of identities that are allowed access through
      [IngressPolicy]. Identities can be an individual user, service account,
      Google group, or third-party identity. The `v1` identities that have the
      prefix `user`, `group`, `serviceAccount`, `principal`, and
      `principalSet` in https://cloud.google.com/iam/docs/principal-
      identifiers#v1 are supported.
    identityType: Specifies the type of identities that are allowed access
      from outside the perimeter. If left unspecified, then members of
      `identities` field will be allowed access.
    sources: Sources that this IngressPolicy authorizes access from.
  """

    class IdentityTypeValueValuesEnum(_messages.Enum):
        """Specifies the type of identities that are allowed access from outside
    the perimeter. If left unspecified, then members of `identities` field
    will be allowed access.

    Values:
      IDENTITY_TYPE_UNSPECIFIED: No blanket identity group specified.
      ANY_IDENTITY: Authorize access from all identities outside the
        perimeter.
      ANY_USER_ACCOUNT: Authorize access from all human users outside the
        perimeter.
      ANY_SERVICE_ACCOUNT: Authorize access from all service accounts outside
        the perimeter.
    """
        IDENTITY_TYPE_UNSPECIFIED = 0
        ANY_IDENTITY = 1
        ANY_USER_ACCOUNT = 2
        ANY_SERVICE_ACCOUNT = 3
    identities = _messages.StringField(1, repeated=True)
    identityType = _messages.EnumField('IdentityTypeValueValuesEnum', 2)
    sources = _messages.MessageField('GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfigIngressSource', 3, repeated=True)