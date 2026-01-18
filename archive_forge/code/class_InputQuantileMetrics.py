from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InputQuantileMetrics(_messages.Message):
    """A InputQuantileMetrics object.

  Fields:
    bytesRead: A Quantiles attribute.
    recordsRead: A Quantiles attribute.
  """
    bytesRead = _messages.MessageField('Quantiles', 1)
    recordsRead = _messages.MessageField('Quantiles', 2)