from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IterationResult(_messages.Message):
    """Information about a single iteration of the training run.

  Fields:
    arimaResult: Arima result.
    clusterInfos: Information about top clusters for clustering models.
    durationMs: Time taken to run the iteration in milliseconds.
    evalLoss: Loss computed on the eval data at the end of iteration.
    index: Index of the iteration, 0 based.
    learnRate: Learn rate used for this iteration.
    principalComponentInfos: The information of the principal components.
    trainingLoss: Loss computed on the training data at the end of iteration.
  """
    arimaResult = _messages.MessageField('ArimaResult', 1)
    clusterInfos = _messages.MessageField('ClusterInfo', 2, repeated=True)
    durationMs = _messages.IntegerField(3)
    evalLoss = _messages.FloatField(4)
    index = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    learnRate = _messages.FloatField(6)
    principalComponentInfos = _messages.MessageField('PrincipalComponentInfo', 7, repeated=True)
    trainingLoss = _messages.FloatField(8)