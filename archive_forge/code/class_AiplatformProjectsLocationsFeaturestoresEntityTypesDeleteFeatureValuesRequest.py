from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteFeatureValuesRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresEntityTypesDeleteFeatureValues
  Request object.

  Fields:
    entityType: Required. The resource name of the EntityType grouping the
      Features for which values are being deleted from. Format: `projects/{pro
      ject}/locations/{location}/featurestores/{featurestore}/entityTypes/{ent
      ityType}`
    googleCloudAiplatformV1DeleteFeatureValuesRequest: A
      GoogleCloudAiplatformV1DeleteFeatureValuesRequest resource to be passed
      as the request body.
  """
    entityType = _messages.StringField(1, required=True)
    googleCloudAiplatformV1DeleteFeatureValuesRequest = _messages.MessageField('GoogleCloudAiplatformV1DeleteFeatureValuesRequest', 2)