from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1Trial(_messages.Message):
    """A message representing a trial.

  Enums:
    StateValueValuesEnum: The detailed state of a trial.

  Fields:
    clientId: Output only. The identifier of the client that originally
      requested this trial.
    endTime: Output only. Time at which the trial's status changed to
      COMPLETED.
    finalMeasurement: The final measurement containing the objective value.
    infeasibleReason: Output only. A human readable string describing why the
      trial is infeasible. This should only be set if trial_infeasible is
      true.
    measurements: A list of measurements that are strictly lexicographically
      ordered by their induced tuples (steps, elapsed_time). These are used
      for early stopping computations.
    name: Output only. Name of the trial assigned by the service.
    parameters: The parameters of the trial.
    startTime: Output only. Time at which the trial was started.
    state: The detailed state of a trial.
    trialInfeasible: Output only. If true, the parameters in this trial are
      not attempted again.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The detailed state of a trial.

    Values:
      STATE_UNSPECIFIED: The trial state is unspecified.
      REQUESTED: Indicates that a specific trial has been requested, but it
        has not yet been suggested by the service.
      ACTIVE: Indicates that the trial has been suggested.
      COMPLETED: Indicates that the trial is done, and either has a
        final_measurement set, or is marked as trial_infeasible.
      STOPPING: Indicates that the trial should stop according to the service.
    """
        STATE_UNSPECIFIED = 0
        REQUESTED = 1
        ACTIVE = 2
        COMPLETED = 3
        STOPPING = 4
    clientId = _messages.StringField(1)
    endTime = _messages.StringField(2)
    finalMeasurement = _messages.MessageField('GoogleCloudMlV1Measurement', 3)
    infeasibleReason = _messages.StringField(4)
    measurements = _messages.MessageField('GoogleCloudMlV1Measurement', 5, repeated=True)
    name = _messages.StringField(6)
    parameters = _messages.MessageField('GoogleCloudMlV1TrialParameter', 7, repeated=True)
    startTime = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    trialInfeasible = _messages.BooleanField(10)