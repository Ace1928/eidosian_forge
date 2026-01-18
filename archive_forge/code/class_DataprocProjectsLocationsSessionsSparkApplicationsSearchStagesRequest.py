from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsSparkApplicationsSearchStagesRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsSparkApplicationsSearchStagesRequest
  object.

  Enums:
    StageStatusValueValuesEnum: Optional. List only stages in the given state.

  Fields:
    name: Required. The fully qualified name of the session to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/sessions/SESSION_I
      D/sparkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of stages (paging based on stage_id) to
      return in each response. The service may return fewer than this. The
      default page size is 10; the maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      SearchSessionSparkApplicationStages call. Provide this token to retrieve
      the subsequent page.
    parent: Required. Parent (Session) resource reference.
    stageStatus: Optional. List only stages in the given state.
    summaryMetricsMask: Optional. The list of summary metrics fields to
      include. Empty list will default to skip all summary metrics fields.
      Example, if the response should include TaskQuantileMetrics, the request
      should have task_quantile_metrics in summary_metrics_mask field
  """

    class StageStatusValueValuesEnum(_messages.Enum):
        """Optional. List only stages in the given state.

    Values:
      STAGE_STATUS_UNSPECIFIED: <no description>
      STAGE_STATUS_ACTIVE: <no description>
      STAGE_STATUS_COMPLETE: <no description>
      STAGE_STATUS_FAILED: <no description>
      STAGE_STATUS_PENDING: <no description>
      STAGE_STATUS_SKIPPED: <no description>
    """
        STAGE_STATUS_UNSPECIFIED = 0
        STAGE_STATUS_ACTIVE = 1
        STAGE_STATUS_COMPLETE = 2
        STAGE_STATUS_FAILED = 3
        STAGE_STATUS_PENDING = 4
        STAGE_STATUS_SKIPPED = 5
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4)
    stageStatus = _messages.EnumField('StageStatusValueValuesEnum', 5)
    summaryMetricsMask = _messages.StringField(6)