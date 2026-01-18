from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BqmlTrainingRun(_messages.Message):
    """A BqmlTrainingRun object.

  Messages:
    TrainingOptionsValue: Deprecated.

  Fields:
    iterationResults: Deprecated.
    startTime: Deprecated.
    state: Deprecated.
    trainingOptions: Deprecated.
  """

    class TrainingOptionsValue(_messages.Message):
        """Deprecated.

    Fields:
      earlyStop: A boolean attribute.
      l1Reg: A number attribute.
      l2Reg: A number attribute.
      learnRate: A number attribute.
      learnRateStrategy: A string attribute.
      lineSearchInitLearnRate: A number attribute.
      maxIteration: A string attribute.
      minRelProgress: A number attribute.
      warmStart: A boolean attribute.
    """
        earlyStop = _messages.BooleanField(1)
        l1Reg = _messages.FloatField(2)
        l2Reg = _messages.FloatField(3)
        learnRate = _messages.FloatField(4)
        learnRateStrategy = _messages.StringField(5)
        lineSearchInitLearnRate = _messages.FloatField(6)
        maxIteration = _messages.IntegerField(7)
        minRelProgress = _messages.FloatField(8)
        warmStart = _messages.BooleanField(9)
    iterationResults = _messages.MessageField('BqmlIterationResult', 1, repeated=True)
    startTime = _message_types.DateTimeField(2)
    state = _messages.StringField(3)
    trainingOptions = _messages.MessageField('TrainingOptionsValue', 4)