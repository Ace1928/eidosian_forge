from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigOperationsListRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigOperationsListRequest object.

  Fields:
    filter: An expression that filters the list of returned operations. A
      filter expression consists of a field name, a comparison operator, and a
      value for filtering. The value must be a string, a number, or a boolean.
      The comparison operator must be one of: `<`, `>`, `<=`, `>=`, `!=`, `=`,
      or `:`. Colon `:` is the contains operator. Filter rules are not case
      sensitive. The following fields in the Operation are eligible for
      filtering: * `name` - The name of the long-running operation * `done` -
      False if the operation is in progress, else true. * `metadata.@type` -
      the type of metadata. For example, the type string for
      CreateInstanceConfigMetadata is `type.googleapis.com/google.spanner.admi
      n.instance.v1.CreateInstanceConfigMetadata`. * `metadata.` - any field
      in metadata.value. `metadata.@type` must be specified first, if
      filtering on metadata fields. * `error` - Error associated with the
      long-running operation. * `response.@type` - the type of response. *
      `response.` - any field in response.value. You can combine multiple
      expressions by enclosing each expression in parentheses. By default,
      expressions are combined with AND logic. However, you can specify AND,
      OR, and NOT logic explicitly. Here are a few examples: * `done:true` -
      The operation is complete. * `(metadata.@type=` \\ `type.googleapis.com/g
      oogle.spanner.admin.instance.v1.CreateInstanceConfigMetadata) AND` \\
      `(metadata.instance_config.name:custom-config) AND` \\
      `(metadata.progress.start_time < \\"2021-03-28T14:50:00Z\\") AND` \\
      `(error:*)` - Return operations where: * The operation's metadata type
      is CreateInstanceConfigMetadata. * The instance config name contains
      "custom-config". * The operation started before 2021-03-28T14:50:00Z. *
      The operation resulted in an error.
    pageSize: Number of operations to be returned in the response. If 0 or
      less, defaults to the server's maximum allowed page size.
    pageToken: If non-empty, `page_token` should contain a next_page_token
      from a previous ListInstanceConfigOperationsResponse to the same
      `parent` and with the same `filter`.
    parent: Required. The project of the instance config operations. Values
      are of the form `projects/`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)