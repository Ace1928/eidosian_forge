from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsConversionWorkspacesSearchBackgroundJobsRequest(_messages.Message):
    """A DatamigrationProjectsLocationsConversionWorkspacesSearchBackgroundJobs
  Request object.

  Fields:
    completedUntilTime: Optional. If provided, only returns jobs that
      completed until (not including) the given timestamp.
    conversionWorkspace: Required. Name of the conversion workspace resource
      whose jobs are listed, in the form of: projects/{project}/locations/{loc
      ation}/conversionWorkspaces/{conversion_workspace}.
    maxSize: Optional. The maximum number of jobs to return. The service may
      return fewer than this value. If unspecified, at most 100 jobs are
      returned. The maximum value is 100; values above 100 are coerced to 100.
    returnMostRecentPerJobType: Optional. Whether or not to return just the
      most recent job per job type,
  """
    completedUntilTime = _messages.StringField(1)
    conversionWorkspace = _messages.StringField(2, required=True)
    maxSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    returnMostRecentPerJobType = _messages.BooleanField(4)