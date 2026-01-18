from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsListRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsListRequest object.

  Fields:
    filter: A filter expression that filters migration jobs listed in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be either =, !=, >, or <. For example, list migration jobs created this
      year by specifying **createTime %gt; 2020-01-01T00:00:00.000000000Z.**
      You can also filter nested fields. For example, you could specify
      **reverseSshConnectivity.vmIp = "1.2.3.4"** to select all migration jobs
      connecting through the specific SSH tunnel bastion.
    orderBy: Sort the results based on the migration job name. Valid values
      are: "name", "name asc", and "name desc".
    pageSize: The maximum number of migration jobs to return. The service may
      return fewer than this value. If unspecified, at most 50 migration jobs
      will be returned. The maximum value is 1000; values above 1000 will be
      coerced to 1000.
    pageToken: The nextPageToken value received in the previous call to
      migrationJobs.list, used in the subsequent request to retrieve the next
      page of results. On first call this should be left blank. When
      paginating, all other parameters provided to migrationJobs.list must
      match the call that provided the page token.
    parent: Required. The parent, which owns this collection of migrationJobs.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)