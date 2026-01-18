from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerOrganizationsGcpUserAccessBindingsDeleteRequest(_messages.Message):
    """A AccesscontextmanagerOrganizationsGcpUserAccessBindingsDeleteRequest
  object.

  Fields:
    name: Required. Example:
      "organizations/256/gcpUserAccessBindings/b3-BhcX_Ud5N"
  """
    name = _messages.StringField(1, required=True)