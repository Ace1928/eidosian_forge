from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorsRequest(_messages.Message):
    """A
  DataprocProjectsLocationsSessionsSparkApplicationsSearchExecutorsRequest
  object.

  Enums:
    ExecutorStatusValueValuesEnum: Optional. Filter to select whether active/
      dead or all executors should be selected.

  Fields:
    executorStatus: Optional. Filter to select whether active/ dead or all
      executors should be selected.
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of executors to return in each
      response. The service may return fewer than this. The default page size
      is 10; the maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplicationExecutors call. Provide this token to
      retrieve the subsequent page.
    parent: Required. Parent (Session) resource reference.
  """

    class ExecutorStatusValueValuesEnum(_messages.Enum):
        """Optional. Filter to select whether active/ dead or all executors
    should be selected.

    Values:
      EXECUTOR_STATUS_UNSPECIFIED: <no description>
      EXECUTOR_STATUS_ACTIVE: <no description>
      EXECUTOR_STATUS_DEAD: <no description>
    """
        EXECUTOR_STATUS_UNSPECIFIED = 0
        EXECUTOR_STATUS_ACTIVE = 1
        EXECUTOR_STATUS_DEAD = 2
    executorStatus = _messages.EnumField('ExecutorStatusValueValuesEnum', 1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5)