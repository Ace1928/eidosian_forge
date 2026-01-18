from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultValues(_messages.Message):
    """DefaultValues represents the default configuration values.

  Fields:
    machineType: Output only. The default machine type used by the backend if
      not provided by the user.
  """
    machineType = _messages.StringField(1)