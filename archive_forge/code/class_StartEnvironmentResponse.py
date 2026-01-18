from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartEnvironmentResponse(_messages.Message):
    """Message included in the response field of operations returned from
  StartEnvironment once the operation is complete.

  Fields:
    environment: Environment that was started.
  """
    environment = _messages.MessageField('Environment', 1)