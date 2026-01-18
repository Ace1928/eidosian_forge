from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ReadFeatureValuesRequest(_messages.Message):
    """Request message for FeaturestoreOnlineServingService.ReadFeatureValues.

  Fields:
    entityId: Required. ID for a specific entity. For example, for a machine
      learning model predicting user clicks on a website, an entity ID could
      be `user_123`.
    featureSelector: Required. Selector choosing Features of the target
      EntityType.
  """
    entityId = _messages.StringField(1)
    featureSelector = _messages.MessageField('GoogleCloudAiplatformV1FeatureSelector', 2)