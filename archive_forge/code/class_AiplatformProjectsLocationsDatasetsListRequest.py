from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsListRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsListRequest object.

  Fields:
    filter: An expression for filtering the results of the request. For field
      names both snake_case and camelCase are supported. * `display_name`:
      supports = and != * `metadata_schema_uri`: supports = and != * `labels`
      supports general map functions that is: * `labels.key=value` - key:value
      equality * `labels.key:* or labels:key - key existence * A key including
      a space must be quoted. `labels."a key"`. Some examples: *
      `displayName="myDisplayName"` * `labels.myKey="myValue"`
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending. Supported fields: *
      `display_name` * `create_time` * `update_time`
    pageSize: The standard list page size.
    pageToken: The standard list page token.
    parent: Required. The name of the Dataset's parent resource. Format:
      `projects/{project}/locations/{location}`
    readMask: Mask specifying which fields to read.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    readMask = _messages.StringField(6)