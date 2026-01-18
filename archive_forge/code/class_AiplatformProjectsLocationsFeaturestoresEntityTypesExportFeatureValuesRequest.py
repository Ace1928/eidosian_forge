from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesExportFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesExportFeatureValues
  Request object.

  Fields:
    entityType: Required. The resource name of the EntityType from which to
      export Feature values. Format: `projects/{project}/locations/{location}/
      featurestores/{featurestore}/entityTypes/{entity_type}`
    googleCloudAiplatformV1ExportFeatureValuesRequest: A
      GoogleCloudAiplatformV1ExportFeatureValuesRequest resource to be passed
      as the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1ExportFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1ExportFeatureValuesRequest', 2)