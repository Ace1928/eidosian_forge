from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchUpdateEntityTypesRequest(_messages.Message):
    """The request message for EntityTypes.BatchUpdateEntityTypes.

  Fields:
    entityTypeBatchInline: The collection of entity types to update or create.
    entityTypeBatchUri: The URI to a Google Cloud Storage file containing
      entity types to update or create. The file format can either be a
      serialized proto (of EntityBatch type) or a JSON object. Note: The URI
      must start with "gs://".
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    updateMask: Optional. The mask to control which fields get updated.
  """
    entityTypeBatchInline = _messages.MessageField('GoogleCloudDialogflowV2EntityTypeBatch', 1)
    entityTypeBatchUri = _messages.StringField(2)
    languageCode = _messages.StringField(3)
    updateMask = _messages.StringField(4)