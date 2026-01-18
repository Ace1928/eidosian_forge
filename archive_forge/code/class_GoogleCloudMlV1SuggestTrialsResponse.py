from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1SuggestTrialsResponse(_messages.Message):
    """This message will be placed in the response field of a completed
  google.longrunning.Operation associated with a SuggestTrials request.

  Enums:
    StudyStateValueValuesEnum: The state of the study.

  Fields:
    endTime: The time at which operation processing completed.
    startTime: The time at which the operation was started.
    studyState: The state of the study.
    trials: A list of trials.
  """

    class StudyStateValueValuesEnum(_messages.Enum):
        """The state of the study.

    Values:
      STATE_UNSPECIFIED: The study state is unspecified.
      ACTIVE: The study is active.
      INACTIVE: The study is stopped due to an internal error.
      COMPLETED: The study is done when the service exhausts the parameter
        search space or max_trial_count is reached.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        INACTIVE = 2
        COMPLETED = 3
    endTime = _messages.StringField(1)
    startTime = _messages.StringField(2)
    studyState = _messages.EnumField('StudyStateValueValuesEnum', 3)
    trials = _messages.MessageField('GoogleCloudMlV1Trial', 4, repeated=True)