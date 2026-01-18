from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchCreateEntitiesRequest(_messages.Message):
    """The request message for EntityTypes.BatchCreateEntities.

  Fields:
    entities: Required. The entities to create.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
  """
    entities = _messages.MessageField('GoogleCloudDialogflowV2EntityTypeEntity', 1, repeated=True)
    languageCode = _messages.StringField(2)