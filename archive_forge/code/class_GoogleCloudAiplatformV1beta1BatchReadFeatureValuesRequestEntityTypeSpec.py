from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchReadFeatureValuesRequestEntityTypeSpec(_messages.Message):
    """Selects Features of an EntityType to read values of and specifies read
  settings.

  Fields:
    entityTypeId: Required. ID of the EntityType to select Features. The
      EntityType id is the entity_type_id specified during EntityType
      creation.
    featureSelector: Required. Selectors choosing which Feature values to read
      from the EntityType.
    settings: Per-Feature settings for the batch read.
  """
    entityTypeId = _messages.StringField(1)
    featureSelector = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureSelector', 2)
    settings = _messages.MessageField('GoogleCloudAiplatformV1beta1DestinationFeatureSetting', 3, repeated=True)