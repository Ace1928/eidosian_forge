from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsJobsExecutionsListRequest(_messages.Message):
    """A RunProjectsLocationsJobsExecutionsListRequest object.

  Fields:
    pageSize: Maximum number of Executions to return in this call.
    pageToken: A page token received from a previous call to ListExecutions.
      All other parameters must match.
    parent: Required. The Execution from which the Executions should be
      listed. To list all Executions across Jobs, use "-" instead of Job name.
      Format: `projects/{project}/locations/{location}/jobs/{job}`, where
      `{project}` can be project id or number.
    showDeleted: If true, returns deleted (but unexpired) resources along with
      active ones.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)