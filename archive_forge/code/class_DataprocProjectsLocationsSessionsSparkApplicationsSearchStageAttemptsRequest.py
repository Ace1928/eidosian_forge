from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptsRequest(_messages.Message):
    """A
  DataprocProjectsLocationsSessionsSparkApplicationsSearchStageAttemptsRequest
  object.

  Fields:
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of stage attempts (paging based on
      stage_attempt_id) to return in each response. The service may return
      fewer than this. The default page size is 10; the maximum page size is
      100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplicationStageAttempts call. Provide this token to
      retrieve the subsequent page.
    parent: Required. Parent (Session) resource reference.
    stageId: Required. Stage ID for which attempts are to be fetched
    summaryMetricsMask: Optional. The list of summary metrics fields to
      include. Empty list will default to skip all summary metrics fields.
      Example, if the response should include TaskQuantileMetrics, the request
      should have task_quantile_metrics in summary_metrics_mask field
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4)
    stageId = _messages.IntegerField(5)
    summaryMetricsMask = _messages.StringField(6)