from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudschedulerProjectsLocationsJobsPatchRequest(_messages.Message):
    """A CloudschedulerProjectsLocationsJobsPatchRequest object.

  Fields:
    job: A Job resource to be passed as the request body.
    name: Optionally caller-specified in CreateJob, after which it becomes
      output only.  The job name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/jobs/JOB_ID`.  * `PROJECT_ID`
      can contain letters ([A-Za-z]), numbers ([0-9]),    hyphens (-), colons
      (:), or periods (.).    For more information, see    [Identifying
      projects](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects#identifying_projects) * `LOCATION_ID` is the canonical
      ID for the job's location.    The list of available locations can be
      obtained by calling    ListLocations.    For more information, see
      https://cloud.google.com/about/locations/. * `JOB_ID` can contain only
      letters ([A-Za-z]), numbers ([0-9]),    hyphens (-), or underscores (_).
      The maximum length is 500 characters.
    updateMask: A  mask used to specify which fields of the job are being
      updated.
  """
    job = _messages.MessageField('Job', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)