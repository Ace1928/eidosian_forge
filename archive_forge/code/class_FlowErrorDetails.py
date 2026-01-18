from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlowErrorDetails(_messages.Message):
    """Encapsulation of flow-specific error details for debugging. Used as a
  details field on an error Status, not intended for external use.

  Fields:
    exceptionType: The type of exception (as a class name).
    flowStepId: The step that failed.
  """
    exceptionType = _messages.StringField(1)
    flowStepId = _messages.StringField(2)