from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchDeletePipelineJobsRequest(_messages.Message):
    """Request message for PipelineService.BatchDeletePipelineJobs.

  Fields:
    names: Required. The names of the PipelineJobs to delete. A maximum of 32
      PipelineJobs can be deleted in a batch. Format:
      `projects/{project}/locations/{location}/pipelineJobs/{pipelineJob}`
  """
    names = _messages.StringField(1, repeated=True)