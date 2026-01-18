from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksJobsCancelRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksJobsCancelRequest object.

  Fields:
    googleCloudDataplexV1CancelJobRequest: A
      GoogleCloudDataplexV1CancelJobRequest resource to be passed as the
      request body.
    name: Required. The resource name of the job: projects/{project_number}/lo
      cations/{location_id}/lakes/{lake_id}/task/{task_id}/job/{job_id}.
  """
    googleCloudDataplexV1CancelJobRequest = _messages.MessageField('GoogleCloudDataplexV1CancelJobRequest', 1)
    name = _messages.StringField(2, required=True)