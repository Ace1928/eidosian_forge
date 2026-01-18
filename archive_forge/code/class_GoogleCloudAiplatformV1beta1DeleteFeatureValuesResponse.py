from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1DeleteFeatureValuesResponse(_messages.Message):
    """Response message for FeaturestoreService.DeleteFeatureValues.

  Fields:
    selectEntity: Response for request specifying the entities to delete
    selectTimeRangeAndFeature: Response for request specifying time range and
      feature
  """
    selectEntity = _messages.MessageField('GoogleCloudAiplatformV1beta1DeleteFeatureValuesResponseSelectEntity', 1)
    selectTimeRangeAndFeature = _messages.MessageField('GoogleCloudAiplatformV1beta1DeleteFeatureValuesResponseSelectTimeRangeAndFeature', 2)