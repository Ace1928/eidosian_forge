from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2BatchUpdateEntitiesRequest(_messages.Message):
    """The request message for EntityTypes.BatchUpdateEntities.

  Fields:
    entities: Required. The entities to update or create.
    languageCode: Optional. The language used to access language-specific
      data. If not specified, the agent's default language is used. For more
      information, see [Multilingual intent and entity
      data](https://cloud.google.com/dialogflow/docs/agents-
      multilingual#intent-entity).
    updateMask: Optional. The mask to control which fields get updated.
  """
    entities = _messages.MessageField('GoogleCloudDialogflowV2EntityTypeEntity', 1, repeated=True)
    languageCode = _messages.StringField(2)
    updateMask = _messages.StringField(3)