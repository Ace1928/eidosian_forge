from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsGetRequest(_messages.Message):
    """A DataflowProjectsJobsGetRequest object.

  Enums:
    ViewValueValuesEnum: The level of information requested in response.

  Fields:
    jobId: The job ID.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains this job.
    projectId: The ID of the Cloud Platform project that the job belongs to.
    view: The level of information requested in response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The level of information requested in response.

    Values:
      JOB_VIEW_UNKNOWN: The job view to return isn't specified, or is unknown.
        Responses will contain at least the `JOB_VIEW_SUMMARY` information,
        and may contain additional information.
      JOB_VIEW_SUMMARY: Request summary information only: Project ID, Job ID,
        job name, job type, job status, start/end time, and Cloud SDK version
        details.
      JOB_VIEW_ALL: Request all information available for this job. When the
        job is in `JOB_STATE_PENDING`, the job has been created but is not yet
        running, and not all job information is available. For complete job
        information, wait until the job in is `JOB_STATE_RUNNING`. For more
        information, see [JobState](https://cloud.google.com/dataflow/docs/ref
        erence/rest/v1b3/projects.jobs#jobstate).
      JOB_VIEW_DESCRIPTION: Request summary info and limited job description
        data for steps, labels and environment.
    """
        JOB_VIEW_UNKNOWN = 0
        JOB_VIEW_SUMMARY = 1
        JOB_VIEW_ALL = 2
        JOB_VIEW_DESCRIPTION = 3
    jobId = _messages.StringField(1, required=True)
    location = _messages.StringField(2)
    projectId = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)