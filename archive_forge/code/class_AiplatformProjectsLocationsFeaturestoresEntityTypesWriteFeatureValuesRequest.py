from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesWriteFeatureValuesRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeaturestoresEntityTypesWriteFeatureValuesRequest
  object.

  Fields:
    entityType: Required. The resource name of the EntityType for the entities
      being written. Value format:
      `projects/{project}/locations/{location}/featurestores/
      {featurestore}/entityTypes/{entityType}`. For example, for a machine
      learning model predicting user clicks on a website, an EntityType ID
      could be `user`.
    googleCloudAiplatformV1WriteFeatureValuesRequest: A
      GoogleCloudAiplatformV1WriteFeatureValuesRequest resource to be passed
      as the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1WriteFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1WriteFeatureValuesRequest', 2)