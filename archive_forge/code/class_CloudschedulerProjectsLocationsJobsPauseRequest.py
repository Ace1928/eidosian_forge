from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudschedulerProjectsLocationsJobsPauseRequest(_messages.Message):
    """A CloudschedulerProjectsLocationsJobsPauseRequest object.

  Fields:
    name: Required.  The job name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/jobs/JOB_ID`.
    pauseJobRequest: A PauseJobRequest resource to be passed as the request
      body.
  """
    name = _messages.StringField(1, required=True)
    pauseJobRequest = _messages.MessageField('PauseJobRequest', 2)