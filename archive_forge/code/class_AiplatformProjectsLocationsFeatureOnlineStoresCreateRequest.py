from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresCreateRequest object.

  Fields:
    featureOnlineStoreId: Required. The ID to use for this FeatureOnlineStore,
      which will become the final component of the FeatureOnlineStore's
      resource name. This value may be up to 60 characters, and valid
      characters are `[a-z0-9_]`. The first character cannot be a number. The
      value must be unique within the project and location.
    googleCloudAiplatformV1FeatureOnlineStore: A
      GoogleCloudAiplatformV1FeatureOnlineStore resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to create
      FeatureOnlineStores. Format: `projects/{project}/locations/{location}`
  """
    featureOnlineStoreId = _messages.StringField(1)
    googleCloudAiplatformV1FeatureOnlineStore = _messages.MessageField('GoogleCloudAiplatformV1FeatureOnlineStore', 2)
    parent = _messages.StringField(3, required=True)