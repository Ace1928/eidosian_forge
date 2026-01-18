from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsCreateRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsCreateRequest
  object.

  Fields:
    featureViewId: Required. The ID to use for the FeatureView, which will
      become the final component of the FeatureView's resource name. This
      value may be up to 60 characters, and valid characters are `[a-z0-9_]`.
      The first character cannot be a number. The value must be unique within
      a FeatureOnlineStore.
    googleCloudAiplatformV1FeatureView: A GoogleCloudAiplatformV1FeatureView
      resource to be passed as the request body.
    parent: Required. The resource name of the FeatureOnlineStore to create
      FeatureViews. Format: `projects/{project}/locations/{location}/featureOn
      lineStores/{feature_online_store}`
    runSyncImmediately: Immutable. If set to true, one on demand sync will be
      run immediately, regardless whether the FeatureView.sync_config is
      configured or not.
  """
    featureViewId = _messages.StringField(1)
    googleCloudAiplatformV1FeatureView = _messages.MessageField('GoogleCloudAiplatformV1FeatureView', 2)
    parent = _messages.StringField(3, required=True)
    runSyncImmediately = _messages.BooleanField(4)