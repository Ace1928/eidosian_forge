from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchCancelPipelineJobsRequest(_messages.Message):
    """Request message for PipelineService.BatchCancelPipelineJobs.

  Fields:
    names: Required. The names of the PipelineJobs to cancel. A maximum of 32
      PipelineJobs can be cancelled in a batch. Format:
      `projects/{project}/locations/{location}/pipelineJobs/{pipelineJob}`
  """
    names = _messages.StringField(1, repeated=True)