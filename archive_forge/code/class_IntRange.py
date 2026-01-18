from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntRange(_messages.Message):
    """Range of an int hyperparameter.

  Fields:
    max: Max value of the int parameter.
    min: Min value of the int parameter.
  """
    max = _messages.IntegerField(1)
    min = _messages.IntegerField(2)