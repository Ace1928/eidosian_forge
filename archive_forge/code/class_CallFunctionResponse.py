from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallFunctionResponse(_messages.Message):
    """Response of `CallFunction` method.

  Fields:
    error: Either system or user-function generated error. Set if execution
      was not successful.
    executionId: Execution id of function invocation.
    result: Result populated for successful execution of synchronous function.
      Will not be populated if function does not return a result through
      context.
  """
    error = _messages.StringField(1)
    executionId = _messages.StringField(2)
    result = _messages.StringField(3)