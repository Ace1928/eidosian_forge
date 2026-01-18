from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRuntimeActionSchemasResponse(_messages.Message):
    """Response message for ConnectorsService.ListRuntimeActionSchemas.

  Fields:
    nextPageToken: Next page token.
    runtimeActionSchemas: Runtime action schemas.
  """
    nextPageToken = _messages.StringField(1)
    runtimeActionSchemas = _messages.MessageField('RuntimeActionSchema', 2, repeated=True)