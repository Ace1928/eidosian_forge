from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerFoldersLocationsEntitlementsSearchRequest(_messages.Message):
    """A PrivilegedaccessmanagerFoldersLocationsEntitlementsSearchRequest
  object.

  Enums:
    CallerAccessTypeValueValuesEnum: Required. Only Entitlements where the
      calling user has this access will be returned.

  Fields:
    callerAccessType: Required. Only Entitlements where the calling user has
      this access will be returned.
    filter: Optional. Only Entitlements matching this filter will be returned
      in the response.
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The parent which owns the Entitlement resources.
  """

    class CallerAccessTypeValueValuesEnum(_messages.Enum):
        """Required. Only Entitlements where the calling user has this access
    will be returned.

    Values:
      CALLER_ACCESS_TYPE_UNSPECIFIED: Unspecified access type.
      GRANT_REQUESTER: The user has access to create Grants using this
        Entitlement.
      GRANT_APPROVER: The user has access to approve/deny Grants created under
        this Entitlement.
    """
        CALLER_ACCESS_TYPE_UNSPECIFIED = 0
        GRANT_REQUESTER = 1
        GRANT_APPROVER = 2
    callerAccessType = _messages.EnumField('CallerAccessTypeValueValuesEnum', 1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)