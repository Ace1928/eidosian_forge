from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnvironmentVariable(_messages.Message):
    """A key-value pair passed as an environment variable to the test.

  Fields:
    key: Key for the environment variable.
    value: Value for the environment variable.
  """
    key = _messages.StringField(1)
    value = _messages.StringField(2)