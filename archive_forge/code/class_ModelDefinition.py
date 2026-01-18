from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelDefinition(_messages.Message):
    """A ModelDefinition object.

  Messages:
    ModelOptionsValue: Deprecated.

  Fields:
    modelOptions: Deprecated.
    trainingRuns: Deprecated.
  """

    class ModelOptionsValue(_messages.Message):
        """Deprecated.

    Fields:
      labels: A string attribute.
      lossType: A string attribute.
      modelType: A string attribute.
    """
        labels = _messages.StringField(1, repeated=True)
        lossType = _messages.StringField(2)
        modelType = _messages.StringField(3)
    modelOptions = _messages.MessageField('ModelOptionsValue', 1)
    trainingRuns = _messages.MessageField('BqmlTrainingRun', 2, repeated=True)