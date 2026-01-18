from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DagRun(_messages.Message):
    """A single DAG run.

  Enums:
    StateValueValuesEnum: DAG run state.
    TypeValueValuesEnum: DAG run type (how it got created/executed).

  Fields:
    dagId: The DAG ID of the DAG whose execution is described by this DAG run.
    dagRunId: The DAG run ID.
    endDate: Timestamp when the DAG run ended. Set only if the DAG run has
      finished.
    executionDate: The logical date and time which the DAG run and its task
      instances are running for.
    name: The resource name of the DAG, in the form: "projects/{projectId}/loc
      ations/{locationId}/environments/{environmentId}/dags/{dagId}/dagRuns/{d
      agRunId}".
    startDate: Timestamp when the DAG run started.
    state: DAG run state.
    type: DAG run type (how it got created/executed).
  """

    class StateValueValuesEnum(_messages.Enum):
        """DAG run state.

    Values:
      STATE_UNSPECIFIED: The state of the DAG run is unknown.
      RUNNING: The DAG run is being executed.
      SUCCEEDED: The DAG run is finished successfully.
      FAILED: The DAG run is finished with an error.
      QUEUED: The DAG run is queued for execution.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        SUCCEEDED = 2
        FAILED = 3
        QUEUED = 4

    class TypeValueValuesEnum(_messages.Enum):
        """DAG run type (how it got created/executed).

    Values:
      TYPE_UNSPECIFIED: The type of the DAG run is unknown.
      BACKFILL: Backfill run.
      SCHEDULED: Scheduled run.
      MANUAL: Manually triggered run.
      DATASET_TRIGGERED: Triggered by a dataset update.
    """
        TYPE_UNSPECIFIED = 0
        BACKFILL = 1
        SCHEDULED = 2
        MANUAL = 3
        DATASET_TRIGGERED = 4
    dagId = _messages.StringField(1)
    dagRunId = _messages.StringField(2)
    endDate = _messages.StringField(3)
    executionDate = _messages.StringField(4)
    name = _messages.StringField(5)
    startDate = _messages.StringField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    type = _messages.EnumField('TypeValueValuesEnum', 8)