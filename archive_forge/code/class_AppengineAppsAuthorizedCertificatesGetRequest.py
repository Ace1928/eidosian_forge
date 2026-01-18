from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsAuthorizedCertificatesGetRequest(_messages.Message):
    """A AppengineAppsAuthorizedCertificatesGetRequest object.

  Enums:
    ViewValueValuesEnum: Controls the set of fields returned in the GET
      response.

  Fields:
    name: Name of the resource requested. Example:
      apps/myapp/authorizedCertificates/12345.
    view: Controls the set of fields returned in the GET response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Controls the set of fields returned in the GET response.

    Values:
      BASIC_CERTIFICATE: Basic certificate information, including applicable
        domains and expiration date.
      FULL_CERTIFICATE: The information from BASIC_CERTIFICATE, plus detailed
        information on the domain mappings that have this certificate mapped.
    """
        BASIC_CERTIFICATE = 0
        FULL_CERTIFICATE = 1
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)