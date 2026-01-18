from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFetchFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsFetchFeature
  ValuesRequest object.

  Fields:
    featureView: Required. FeatureView resource format `projects/{project}/loc
      ations/{location}/featureOnlineStores/{featureOnlineStore}/featureViews/
      {featureView}`
    googleCloudAiplatformV1FetchFeatureValuesRequest: A
      GoogleCloudAiplatformV1FetchFeatureValuesRequest resource to be passed
      as the request body.
  """
    featureView = _messages.StringField(1, required=True)
    googleCloudAiplatformV1FetchFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1FetchFeatureValuesRequest', 2)