from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSyncRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsSyncRequest
  object.

  Fields:
    featureView: Required. Format: `projects/{project}/locations/{location}/fe
      atureOnlineStores/{feature_online_store}/featureViews/{feature_view}`
    googleCloudAiplatformV1beta1SyncFeatureViewRequest: A
      GoogleCloudAiplatformV1beta1SyncFeatureViewRequest resource to be passed
      as the request body.
  """
    featureView = _messages.StringField(1, required=True)
    googleCloudAiplatformV1beta1SyncFeatureViewRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1SyncFeatureViewRequest', 2)