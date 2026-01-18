from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesUpsertDatapointsRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesUpsertDatapointsRequest object.

  Fields:
    googleCloudAiplatformV1UpsertDatapointsRequest: A
      GoogleCloudAiplatformV1UpsertDatapointsRequest resource to be passed as
      the request body.
    index: Required. The name of the Index resource to be updated. Format:
      `projects/{project}/locations/{location}/indexes/{index}`
  """
    googleCloudAiplatformV1UpsertDatapointsRequest = _messages.MessageField('GoogleCloudAiplatformV1UpsertDatapointsRequest', 1)
    index = _messages.StringField(2, required=True)