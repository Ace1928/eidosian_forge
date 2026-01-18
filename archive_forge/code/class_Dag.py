from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Dag(_messages.Message):
    """A Composer DAG resource.

  Enums:
    CatchupValueValuesEnum: Whether the catchup is enabled for the DAG.
    StateValueValuesEnum: Output only. The current state of the DAG.

  Fields:
    catchup: Whether the catchup is enabled for the DAG.
    cronSchedule: The DAG's schedule in cron format.
    dagId: Required. The DAG ID.
    dagrunTimeout: Maximum runtime of a DAG run before termination with a
      timeout.
    description: The description of the DAG.
    durationSchedule: The DAG's schedule as a time duration between runs.
    endDate: The end_date parameter of the DAG (if set).
    failStop: Whether a "fail_stop" mode is enabled for the DAG.
    fileloc: File location relative to the Cloud Storage bucket root folder.
    lastRunEndTime: The end timestamp of the last completed DAG run.
    lastUpdated: The last time the DAG has been serialized.
    maxActiveRuns: Maximum number of simultaneous active runs of the DAG.
      Default is zero.
    maxActiveTasks: Maximum number of simultaneous active tasks of the DAG.
      Default is zero.
    name: Required. The resource name of the DAG, in the form: "projects/{proj
      ectId}/locations/{locationId}/environments/{environmentId}/dags/{dagId}"
      .
    runningCount: The number of running instances of the DAG.
    startDate: The start_date parameter of the DAG.
    state: Output only. The current state of the DAG.
  """

    class CatchupValueValuesEnum(_messages.Enum):
        """Whether the catchup is enabled for the DAG.

    Values:
      CATCHUP_VALUE_UNSPECIFIED: The state of the Cachup is unknown.
      ENABLED: The catchup is enabled.
      DISABLED: The catchup is disabled.
    """
        CATCHUP_VALUE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the DAG.

    Values:
      STATE_UNSPECIFIED: The state of the DAG is unknown.
      ACTIVE: The DAG is available for execution.
      PAUSED: The DAG is paused.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PAUSED = 2
    catchup = _messages.EnumField('CatchupValueValuesEnum', 1)
    cronSchedule = _messages.StringField(2)
    dagId = _messages.StringField(3)
    dagrunTimeout = _messages.StringField(4)
    description = _messages.StringField(5)
    durationSchedule = _messages.StringField(6)
    endDate = _messages.StringField(7)
    failStop = _messages.BooleanField(8)
    fileloc = _messages.StringField(9)
    lastRunEndTime = _messages.StringField(10)
    lastUpdated = _messages.StringField(11)
    maxActiveRuns = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    maxActiveTasks = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    name = _messages.StringField(14)
    runningCount = _messages.IntegerField(15, variant=_messages.Variant.INT32)
    startDate = _messages.StringField(16)
    state = _messages.EnumField('StateValueValuesEnum', 17)