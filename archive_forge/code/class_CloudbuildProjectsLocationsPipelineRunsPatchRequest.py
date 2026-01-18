from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsPipelineRunsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsPipelineRunsPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, and the PipelineRun is not found,
      a new PipelineRun will be created. In this situation, `update_mask` is
      ignored.
    name: Output only. The `PipelineRun` name with format
      `projects/{project}/locations/{location}/pipelineRuns/{pipeline_run}`
    pipelineRun: A PipelineRun resource to be passed as the request body.
    updateMask: Required. The list of fields to be updated.
    validateOnly: Optional. When true, the query is validated only, but not
      executed.
  """
    allowMissing = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    pipelineRun = _messages.MessageField('PipelineRun', 3)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)