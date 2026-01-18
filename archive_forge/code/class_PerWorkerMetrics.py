from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerWorkerMetrics(_messages.Message):
    """Per worker metrics.

  Fields:
    perStepNamespaceMetrics: Optional. Metrics for a particular unfused step
      and namespace.
  """
    perStepNamespaceMetrics = _messages.MessageField('PerStepNamespaceMetrics', 1, repeated=True)