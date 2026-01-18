from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1betaReplay(_messages.Message):
    """A resource describing a `Replay`, or simulation.

  Enums:
    StateValueValuesEnum: Output only. The current state of the `Replay`.

  Fields:
    config: Required. The configuration used for the `Replay`.
    name: Output only. The resource name of the `Replay`, which has the
      following format: `{projects|folders|organizations}/{resource-
      id}/locations/global/replays/{replay-id}`, where `{resource-id}` is the
      ID of the project, folder, or organization that owns the Replay.
      Example: `projects/my-example-
      project/locations/global/replays/506a5f7f-38ce-4d7d-8e03-479ce1833c36`
    resultsSummary: Output only. Summary statistics about the replayed log
      entries.
    state: Output only. The current state of the `Replay`.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the `Replay`.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      PENDING: The `Replay` has not started yet.
      RUNNING: The `Replay` is currently running.
      SUCCEEDED: The `Replay` has successfully completed.
      FAILED: The `Replay` has finished with an error.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
    config = _messages.MessageField('GoogleCloudPolicysimulatorV1betaReplayConfig', 1)
    name = _messages.StringField(2)
    resultsSummary = _messages.MessageField('GoogleCloudPolicysimulatorV1betaReplayResultsSummary', 3)
    state = _messages.EnumField('StateValueValuesEnum', 4)