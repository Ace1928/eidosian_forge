from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataMaskingStatistics(_messages.Message):
    """Statistics for data-masking.

  Fields:
    dataMaskingApplied: Whether any accessed data was protected by the data
      masking.
  """
    dataMaskingApplied = _messages.BooleanField(1)