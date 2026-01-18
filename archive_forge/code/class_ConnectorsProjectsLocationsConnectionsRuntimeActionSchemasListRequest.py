from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsRuntimeActionSchemasListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsRuntimeActionSchemasListRequest
  object.

  Fields:
    filter: Required. Filter Format: action="{actionId}" Only action field is
      supported with literal equality operator. Accepted filter example:
      action="CancelOrder" Wildcards are not supported in the filter
      currently.
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource of RuntimeActionSchema Format:
      projects/{project}/locations/{location}/connections/{connection}
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)