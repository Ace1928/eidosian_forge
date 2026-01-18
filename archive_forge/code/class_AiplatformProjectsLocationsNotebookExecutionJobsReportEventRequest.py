from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsNotebookExecutionJobsReportEventRequest(_messages.Message):
    """A AiplatformProjectsLocationsNotebookExecutionJobsReportEventRequest
  object.

  Fields:
    googleCloudAiplatformV1beta1ReportExecutionEventRequest: A
      GoogleCloudAiplatformV1beta1ReportExecutionEventRequest resource to be
      passed as the request body.
    name: Required. The name of the NotebookExecutionJob resource. Format: `pr
      ojects/{project}/locations/{location}/notebookExecutionJobs/{notebook_ex
      ecution_jobs}`
  """
    googleCloudAiplatformV1beta1ReportExecutionEventRequest = _messages.MessageField('GoogleCloudAiplatformV1beta1ReportExecutionEventRequest', 1)
    name = _messages.StringField(2, required=True)