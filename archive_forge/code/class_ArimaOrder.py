from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArimaOrder(_messages.Message):
    """Arima order, can be used for both non-seasonal and seasonal parts.

  Fields:
    d: Order of the differencing part.
    p: Order of the autoregressive part.
    q: Order of the moving-average part.
  """
    d = _messages.IntegerField(1)
    p = _messages.IntegerField(2)
    q = _messages.IntegerField(3)