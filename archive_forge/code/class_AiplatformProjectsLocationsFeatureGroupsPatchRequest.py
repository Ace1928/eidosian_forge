from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureGroupsPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureGroupsPatchRequest object.

  Fields:
    googleCloudAiplatformV1FeatureGroup: A GoogleCloudAiplatformV1FeatureGroup
      resource to be passed as the request body.
    name: Identifier. Name of the FeatureGroup. Format:
      `projects/{project}/locations/{location}/featureGroups/{featureGroup}`
    updateMask: Field mask is used to specify the fields to be overwritten in
      the FeatureGroup resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then only the non-empty fields present in the request will be
      overwritten. Set the update_mask to `*` to override all fields.
      Updatable fields: * `labels`
  """
    googleCloudAiplatformV1FeatureGroup = _messages.MessageField('GoogleCloudAiplatformV1FeatureGroup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)