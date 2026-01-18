from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceManagedByIgmErrorManagedInstanceError(_messages.Message):
    """A InstanceManagedByIgmErrorManagedInstanceError object.

  Fields:
    code: [Output Only] Error code.
    message: [Output Only] Error message.
  """
    code = _messages.StringField(1)
    message = _messages.StringField(2)