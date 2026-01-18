from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsPipelineRunsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsPipelineRunsCreateRequest object.

  Fields:
    parent: Required. Value for parent.
    pipelineRun: A PipelineRun resource to be passed as the request body.
    pipelineRunId: Required. The ID to use for the PipelineRun, which will
      become the final component of the PipelineRun's resource name. Names
      must be unique per-project per-location. This value should be 4-63
      characters, and valid characters are /a-z-/.
    validateOnly: Optional. When true, the query is validated only, but not
      executed.
  """
    parent = _messages.StringField(1, required=True)
    pipelineRun = _messages.MessageField('PipelineRun', 2)
    pipelineRunId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)