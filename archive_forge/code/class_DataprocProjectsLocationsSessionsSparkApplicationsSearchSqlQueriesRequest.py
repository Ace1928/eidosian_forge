from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchSqlQueriesRequest(_messages.Message):
    """A
  DataprocProjectsLocationsSessionsSparkApplicationsSearchSqlQueriesRequest
  object.

  Fields:
    details: Optional. Lists/ hides details of Spark plan nodes. True is set
      to list and false to hide.
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of queries to return in each response.
      The service may return fewer than this. The default page size is 10; the
      maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplicationSqlQueries call. Provide this token to
      retrieve the subsequent page.
    parent: Required. Parent (Session) resource reference.
    planDescription: Optional. Enables/ disables physical plan description on
      demand
  """
    details = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5)
    planDescription = _messages.BooleanField(6)