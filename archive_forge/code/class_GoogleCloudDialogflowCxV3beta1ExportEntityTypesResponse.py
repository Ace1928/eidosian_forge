from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ExportEntityTypesResponse(_messages.Message):
    """The response message for EntityTypes.ExportEntityTypes.

  Fields:
    entityTypesContent: Uncompressed byte content for entity types. This field
      is populated only if `entity_types_content_inline` is set to true in
      ExportEntityTypesRequest.
    entityTypesUri: The URI to a file containing the exported entity types.
      This field is populated only if `entity_types_uri` is specified in
      ExportEntityTypesRequest.
  """
    entityTypesContent = _messages.MessageField('GoogleCloudDialogflowCxV3beta1InlineDestination', 1)
    entityTypesUri = _messages.StringField(2)