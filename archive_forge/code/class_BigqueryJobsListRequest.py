from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigqueryJobsListRequest(_messages.Message):
    """A BigqueryJobsListRequest object.

  Enums:
    ProjectionValueValuesEnum: Restrict information returned to a set of
      selected fields
    StateFilterValueValuesEnum: Filter for job state

  Fields:
    allUsers: Whether to display jobs owned by all users in the project.
      Default false
    maxResults: Maximum number of results to return
    pageToken: Page token, returned by a previous call, to request the next
      page of results
    projectId: Project ID of the jobs to list
    projection: Restrict information returned to a set of selected fields
    stateFilter: Filter for job state
  """

    class ProjectionValueValuesEnum(_messages.Enum):
        """Restrict information returned to a set of selected fields

    Values:
      full: Includes all job data
      minimal: Does not include the job configuration
    """
        full = 0
        minimal = 1

    class StateFilterValueValuesEnum(_messages.Enum):
        """Filter for job state

    Values:
      done: Finished jobs
      pending: Pending jobs
      running: Running jobs
    """
        done = 0
        pending = 1
        running = 2
    allUsers = _messages.BooleanField(1)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.UINT32)
    pageToken = _messages.StringField(3)
    projectId = _messages.StringField(4, required=True)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 5)
    stateFilter = _messages.EnumField('StateFilterValueValuesEnum', 6, repeated=True)