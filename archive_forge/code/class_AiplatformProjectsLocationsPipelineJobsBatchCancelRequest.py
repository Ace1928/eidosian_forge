from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPipelineJobsBatchCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsPipelineJobsBatchCancelRequest object.

  Fields:
    googleCloudAiplatformV1BatchCancelPipelineJobsRequest: A
      GoogleCloudAiplatformV1BatchCancelPipelineJobsRequest resource to be
      passed as the request body.
    parent: Required. The name of the PipelineJobs' parent resource. Format:
      `projects/{project}/locations/{location}`
  """
    googleCloudAiplatformV1BatchCancelPipelineJobsRequest = _messages.MessageField('GoogleCloudAiplatformV1BatchCancelPipelineJobsRequest', 1)
    parent = _messages.StringField(2, required=True)