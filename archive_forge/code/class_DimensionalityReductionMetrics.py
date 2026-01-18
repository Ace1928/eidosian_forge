from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DimensionalityReductionMetrics(_messages.Message):
    """Model evaluation metrics for dimensionality reduction models.

  Fields:
    totalExplainedVarianceRatio: Total percentage of variance explained by the
      selected principal components.
  """
    totalExplainedVarianceRatio = _messages.FloatField(1)