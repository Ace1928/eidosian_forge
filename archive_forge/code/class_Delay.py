from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Delay(_messages.Message):
    """message for delay

  Fields:
    fixedDelay: Delay time for requests.
    percentage: Percentage of the network traffic to be delayed.
  """
    fixedDelay = _messages.StringField(1)
    percentage = _messages.IntegerField(2, variant=_messages.Variant.INT32)