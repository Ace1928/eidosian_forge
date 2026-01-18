from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1CreatePipelineJobRequest(_messages.Message):
    """Request message for PipelineService.CreatePipelineJob.

  Fields:
    parent: Required. The resource name of the Location to create the
      PipelineJob in. Format: `projects/{project}/locations/{location}`
    pipelineJob: Required. The PipelineJob to create.
    pipelineJobId: The ID to use for the PipelineJob, which will become the
      final component of the PipelineJob name. If not provided, an ID will be
      automatically generated. This value should be less than 128 characters,
      and valid characters are `/a-z-/`.
  """
    parent = _messages.StringField(1)
    pipelineJob = _messages.MessageField('GoogleCloudAiplatformV1PipelineJob', 2)
    pipelineJobId = _messages.StringField(3)