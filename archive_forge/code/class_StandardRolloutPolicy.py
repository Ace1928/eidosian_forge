from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardRolloutPolicy(_messages.Message):
    """Standard rollout policy is the default policy for blue-green.

  Fields:
    batchNodeCount: Number of blue nodes to drain in a batch.
    batchPercentage: Percentage of the blue pool nodes to drain in a batch.
      The range of this field should be (0.0, 1.0].
    batchSoakDuration: Soak time after each batch gets drained. Default to
      zero.
  """
    batchNodeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    batchPercentage = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    batchSoakDuration = _messages.StringField(3)