from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRuntimeEntitySchemasResponse(_messages.Message):
    """Response message for ConnectorsService.ListRuntimeEntitySchemas.

  Fields:
    nextPageToken: Next page token.
    runtimeEntitySchemas: Runtime entity schemas.
  """
    nextPageToken = _messages.StringField(1)
    runtimeEntitySchemas = _messages.MessageField('RuntimeEntitySchema', 2, repeated=True)