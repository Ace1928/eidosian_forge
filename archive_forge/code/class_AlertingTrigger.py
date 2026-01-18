from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlertingTrigger(_messages.Message):
    """A restriction on the alert test to require a certain count or percent of
  rows to be present.

  Fields:
    count: Optional. The absolute number of time series that must fail the
      predicate for the test to be triggered.
    percent: Optional. The percentage of time series that must fail the
      predicate for the test to be triggered.
  """
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    percent = _messages.FloatField(2)