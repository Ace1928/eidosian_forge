from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesSparkApplicationsSearchExecutorStageSummaryRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesSparkApplicationsSearchExecutorStageSu
  mmaryRequest object.

  Fields:
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID/s
      parkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of executors to return in each
      response. The service may return fewer than this. The default page size
      is 10; the maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      AccessSparkApplicationExecutorsList call. Provide this token to retrieve
      the subsequent page.
    parent: Required. Parent (Batch) resource reference.
    stageAttemptId: Required. Stage Attempt ID
    stageId: Required. Stage ID
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4)
    stageAttemptId = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(6)