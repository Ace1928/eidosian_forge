from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresPatchRequest object.

  Fields:
    googleCloudAiplatformV1Featurestore: A GoogleCloudAiplatformV1Featurestore
      resource to be passed as the request body.
    name: Output only. Name of the Featurestore. Format:
      `projects/{project}/locations/{location}/featurestores/{featurestore}`
    updateMask: Field mask is used to specify the fields to be overwritten in
      the Featurestore resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then only the non-empty fields present in the request will be
      overwritten. Set the update_mask to `*` to override all fields.
      Updatable fields: * `labels` * `online_serving_config.fixed_node_count`
      * `online_serving_config.scaling` * `online_storage_ttl_days`
  """
    googleCloudAiplatformV1Featurestore = _messages.MessageField('GoogleCloudAiplatformV1Featurestore', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)