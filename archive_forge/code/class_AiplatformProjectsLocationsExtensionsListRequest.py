from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsExtensionsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsExtensionsListRequest object.

  Fields:
    filter: Optional. The standard list filter. Supported fields: *
      `display_name` * `create_time` * `update_time` More detail in
      [AIP-160](https://google.aip.dev/160).
    orderBy: Optional. A comma-separated list of fields to order by, sorted in
      ascending order. Use "desc" after a field name for descending. Supported
      fields: * `display_name` * `create_time` * `update_time` Example:
      `display_name, create_time desc`.
    pageSize: Optional. The standard list page size.
    pageToken: Optional. The standard list page token.
    parent: Required. The resource name of the Location to list the Extensions
      from. Format: `projects/{project}/locations/{location}`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)