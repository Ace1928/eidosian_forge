from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresPatchRequest object.

  Fields:
    googleCloudAiplatformV1FeatureOnlineStore: A
      GoogleCloudAiplatformV1FeatureOnlineStore resource to be passed as the
      request body.
    name: Identifier. Name of the FeatureOnlineStore. Format: `projects/{proje
      ct}/locations/{location}/featureOnlineStores/{featureOnlineStore}`
    updateMask: Field mask is used to specify the fields to be overwritten in
      the FeatureOnlineStore resource by the update. The fields specified in
      the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then only the non-empty fields present in the request
      will be overwritten. Set the update_mask to `*` to override all fields.
      Updatable fields: * `big_query_source` * `bigtable` * `labels` *
      `sync_config`
  """
    googleCloudAiplatformV1FeatureOnlineStore = _messages.MessageField('GoogleCloudAiplatformV1FeatureOnlineStore', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)