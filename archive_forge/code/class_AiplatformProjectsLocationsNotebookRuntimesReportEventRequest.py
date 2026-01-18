from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookRuntimesReportEventRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookRuntimesReportEventRequest object.

  Fields:
    googleCloudAiplatformV1beta1ReportRuntimeEventRequest: A
      GoogleCloudAiplatformV1beta1ReportRuntimeEventRequest resource to be
      passed as the request body.
    name: Required. The name of the NotebookRuntime resource. Format: `project
      s/{project}/locations/{location}/notebookRuntimes/{notebook_runtime}`
  """
    googleCloudAiplatformV1beta1ReportRuntimeEventRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ReportRuntimeEventRequest', 1)
    name = _messages.StringField(2, required=True)