from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureOnlineStoresFeatureViewsPatchRequest
  object.

  Fields:
    googleCloudAiplatformV1FeatureView: A GoogleCloudAiplatformV1FeatureView
      resource to be passed as the request body.
    name: Identifier. Name of the FeatureView. Format: `projects/{project}/loc
      ations/{location}/featureOnlineStores/{feature_online_store}/featureView
      s/{feature_view}`
    updateMask: Field mask is used to specify the fields to be overwritten in
      the FeatureView resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then only the non-empty fields present in the request will be
      overwritten. Set the update_mask to `*` to override all fields.
      Updatable fields: * `labels` * `serviceAgentType`
  """
    googleCloudAiplatformV1FeatureView = _messages.MessageField('GoogleCloudAiplatformV1FeatureView', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)