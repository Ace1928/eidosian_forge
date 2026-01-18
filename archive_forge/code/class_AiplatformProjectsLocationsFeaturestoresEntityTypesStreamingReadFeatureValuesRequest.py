from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesStreamingReadFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesStreamingReadFeatur
  eValuesRequest object.

  Fields:
    entityType: Required. The resource name of the entities' type. Value
      format: `projects/{project}/locations/{location}/featurestores/{features
      tore}/entityTypes/{entityType}`. For example, for a machine learning
      model predicting user clicks on a website, an EntityType ID could be
      `user`.
    googleCloudAiplatformV1StreamingReadFeatureValuesRequest: A
      GoogleCloudAiplatformV1StreamingReadFeatureValuesRequest resource to be
      passed as the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1StreamingReadFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1StreamingReadFeatureValuesRequest', 2)