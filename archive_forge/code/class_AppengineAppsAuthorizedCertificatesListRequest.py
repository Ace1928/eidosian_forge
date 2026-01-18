from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsAuthorizedCertificatesListRequest(_messages.Message):
    """A AppengineAppsAuthorizedCertificatesListRequest object.

  Enums:
    ViewValueValuesEnum: Controls the set of fields returned in the LIST
      response.

  Fields:
    pageSize: Maximum results to return per page.
    pageToken: Continuation token for fetching the next page of results.
    parent: Name of the parent Application resource. Example: apps/myapp.
    view: Controls the set of fields returned in the LIST response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Controls the set of fields returned in the LIST response.

    Values:
      BASIC_CERTIFICATE: Basic certificate information, including applicable
        domains and expiration date.
      FULL_CERTIFICATE: The information from BASIC_CERTIFICATE, plus detailed
        information on the domain mappings that have this certificate mapped.
    """
        BASIC_CERTIFICATE = 0
        FULL_CERTIFICATE = 1
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)