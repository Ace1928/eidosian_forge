from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultInjectionAction(_messages.Message):
    """An Action identifies steps to perform in the fault.

  Fields:
    faultInjectionPolicy: FaultInectionPolicy action for fault.
  """
    faultInjectionPolicy = _messages.MessageField('FaultInjectionPolicy', 1)