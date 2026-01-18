from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PipelineJobDetail(_messages.Message):
    """The runtime detail of PipelineJob.

  Fields:
    pipelineContext: Output only. The context of the pipeline.
    pipelineRunContext: Output only. The context of the current pipeline run.
    taskDetails: Output only. The runtime details of the tasks under the
      pipeline.
  """
    pipelineContext = _messages.MessageField('GoogleCloudAiplatformV1Context', 1)
    pipelineRunContext = _messages.MessageField('GoogleCloudAiplatformV1Context', 2)
    taskDetails = _messages.MessageField('GoogleCloudAiplatformV1PipelineTaskDetail', 3, repeated=True)