from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetricShortId(_messages.Message):
    """The metric short id is returned to the user alongside an offset into
  ReportWorkItemStatusRequest

  Fields:
    metricIndex: The index of the corresponding metric in the
      ReportWorkItemStatusRequest. Required.
    shortId: The service-generated short identifier for the metric.
  """
    metricIndex = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    shortId = _messages.IntegerField(2)