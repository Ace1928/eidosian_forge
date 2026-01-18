from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchCreateFeaturesResponse(_messages.Message):
    """Response message for FeaturestoreService.BatchCreateFeatures.

  Fields:
    features: The Features created.
  """
    features = _messages.MessageField('GoogleCloudAiplatformV1Feature', 1, repeated=True)