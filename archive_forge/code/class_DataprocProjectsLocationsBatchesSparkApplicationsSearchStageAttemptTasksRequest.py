from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesSparkApplicationsSearchStageAttemptTasksRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesSparkApplicationsSearchStageAttemptTas
  ksRequest object.

  Enums:
    TaskStatusValueValuesEnum: Optional. List only tasks in the state.

  Fields:
    name: Required. The fully qualified name of the batch to retrieve in the
      format "projects/PROJECT_ID/locations/DATAPROC_REGION/batches/BATCH_ID/s
      parkApplications/APPLICATION_ID"
    pageSize: Optional. Maximum number of tasks to return in each response.
      The service may return fewer than this. The default page size is 10; the
      maximum page size is 100.
    pageToken: Optional. A page token received from a previous
      ListSparkApplicationStageAttemptTasks call. Provide this token to
      retrieve the subsequent page.
    parent: Required. Parent (Batch) resource reference.
    sortRuntime: Optional. Sort the tasks by runtime.
    stageAttemptId: Optional. Stage Attempt ID
    stageId: Optional. Stage ID
    taskStatus: Optional. List only tasks in the state.
  """

    class TaskStatusValueValuesEnum(_messages.Enum):
        """Optional. List only tasks in the state.

    Values:
      TASK_STATUS_UNSPECIFIED: <no description>
      TASK_STATUS_RUNNING: <no description>
      TASK_STATUS_SUCCESS: <no description>
      TASK_STATUS_FAILED: <no description>
      TASK_STATUS_KILLED: <no description>
      TASK_STATUS_PENDING: <no description>
    """
        TASK_STATUS_UNSPECIFIED = 0
        TASK_STATUS_RUNNING = 1
        TASK_STATUS_SUCCESS = 2
        TASK_STATUS_FAILED = 3
        TASK_STATUS_KILLED = 4
        TASK_STATUS_PENDING = 5
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4)
    sortRuntime = _messages.BooleanField(5)
    stageAttemptId = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(7)
    taskStatus = _messages.EnumField('TaskStatusValueValuesEnum', 8)