from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceManagedByIgmError(_messages.Message):
    """A InstanceManagedByIgmError object.

  Fields:
    error: [Output Only] Contents of the error.
    instanceActionDetails: [Output Only] Details of the instance action that
      triggered this error. May be null, if the error was not caused by an
      action on an instance. This field is optional.
    timestamp: [Output Only] The time that this error occurred. This value is
      in RFC3339 text format.
  """
    error = _messages.MessageField('InstanceManagedByIgmErrorManagedInstanceError', 1)
    instanceActionDetails = _messages.MessageField('InstanceManagedByIgmErrorInstanceActionDetails', 2)
    timestamp = _messages.StringField(3)