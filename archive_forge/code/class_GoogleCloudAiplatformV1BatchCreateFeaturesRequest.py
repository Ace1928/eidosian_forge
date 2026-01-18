from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1BatchCreateFeaturesRequest(_messages.Message):
    """Request message for FeaturestoreService.BatchCreateFeatures.

  Fields:
    requests: Required. The request message specifying the Features to create.
      All Features must be created under the same parent EntityType. The
      `parent` field in each child request message can be omitted. If `parent`
      is set in a child request, then the value must match the `parent` value
      in this request message.
  """
    requests = _messages.MessageField('GoogleCloudAiplatformV1CreateFeatureRequest', 1, repeated=True)