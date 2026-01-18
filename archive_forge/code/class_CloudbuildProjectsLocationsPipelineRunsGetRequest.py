from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsPipelineRunsGetRequest(_messages.Message):
    """A CloudbuildProjectsLocationsPipelineRunsGetRequest object.

  Fields:
    name: Required. The name of the PipelineRun to retrieve. Format:
      projects/{project}/locations/{location}/pipelineRuns/{pipelineRun}
  """
    name = _messages.StringField(1, required=True)