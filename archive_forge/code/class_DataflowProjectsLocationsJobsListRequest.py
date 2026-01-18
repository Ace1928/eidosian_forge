from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsJobsListRequest(_messages.Message):
    """A DataflowProjectsLocationsJobsListRequest object.

  Enums:
    FilterValueValuesEnum: The kind of filter to use.
    ViewValueValuesEnum: Deprecated. ListJobs always returns summaries now.
      Use GetJob for other JobViews.

  Fields:
    filter: The kind of filter to use.
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains this job.
    name: Optional. The job name.
    pageSize: If there are many jobs, limit response to at most this many. The
      actual number of jobs returned will be the lesser of max_responses and
      an unspecified server-defined limit.
    pageToken: Set this to the 'next_page_token' field of a previous response
      to request additional results in a long list.
    projectId: The project which owns the jobs.
    view: Deprecated. ListJobs always returns summaries now. Use GetJob for
      other JobViews.
  """

    class FilterValueValuesEnum(_messages.Enum):
        """The kind of filter to use.

    Values:
      UNKNOWN: The filter isn't specified, or is unknown. This returns all
        jobs ordered on descending `JobUuid`.
      ALL: Returns all running jobs first ordered on creation timestamp, then
        returns all terminated jobs ordered on the termination timestamp.
      TERMINATED: Filters the jobs that have a terminated state, ordered on
        the termination timestamp. Example terminated states:
        `JOB_STATE_STOPPED`, `JOB_STATE_UPDATED`, `JOB_STATE_DRAINED`, etc.
      ACTIVE: Filters the jobs that are running ordered on the creation
        timestamp.
    """
        UNKNOWN = 0
        ALL = 1
        TERMINATED = 2
        ACTIVE = 3

    class ViewValueValuesEnum(_messages.Enum):
        """Deprecated. ListJobs always returns summaries now. Use GetJob for
    other JobViews.

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
    filter = _messages.EnumField('FilterValueValuesEnum', 1)
    location = _messages.StringField(2, required=True)
    name = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    projectId = _messages.StringField(6, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 7)