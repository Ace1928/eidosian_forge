from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsExecutionsTasksListRequest(_messages.Message):
    """A RunProjectsLocationsJobsExecutionsTasksListRequest object.

  Fields:
    pageSize: Maximum number of Tasks to return in this call.
    pageToken: A page token received from a previous call to ListTasks. All
      other parameters must match.
    parent: Required. The Execution from which the Tasks should be listed. To
      list all Tasks across Executions of a Job, use "-" instead of Execution
      name. To list all Tasks across Jobs, use "-" instead of Job name.
      Format: projects/{project}/locations/{location}/jobs/{job}/executions/{e
      xecution}
    showDeleted: If true, returns deleted (but unexpired) resources along with
      active ones.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)