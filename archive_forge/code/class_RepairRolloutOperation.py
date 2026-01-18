from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairRolloutOperation(_messages.Message):
    """Contains the information for an automated `repair rollout` operation.

  Fields:
    currentRepairModeIndex: Output only. The index of the current repair
      action in the repair sequence.
    jobId: Output only. The job ID for the Job to repair.
    phaseId: Output only. The phase ID of the phase that includes the job
      being repaired.
    repairPhases: Output only. Records of the repair attempts. Each repair
      phase may have multiple retry attempts or single rollback attempt.
    rollout: Output only. The name of the rollout that initiates the
      `AutomationRun`.
  """
    currentRepairModeIndex = _messages.IntegerField(1)
    jobId = _messages.StringField(2)
    phaseId = _messages.StringField(3)
    repairPhases = _messages.MessageField('RepairPhase', 4, repeated=True)
    rollout = _messages.StringField(5)