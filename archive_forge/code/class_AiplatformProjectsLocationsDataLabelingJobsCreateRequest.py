from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDataLabelingJobsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsDataLabelingJobsCreateRequest object.

  Fields:
    googleCloudAiplatformV1DataLabelingJob: A
      GoogleCloudAiplatformV1DataLabelingJob resource to be passed as the
      request body.
    parent: Required. The parent of the DataLabelingJob. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1DataLabelingJob = _messages.MessageField('GoogleCloudAiplatformV1DataLabelingJob', 1)
    parent = _messages.StringField(2, required=True)