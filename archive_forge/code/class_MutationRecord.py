from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MutationRecord(_messages.Message):
    """Describes a change made to a configuration.

  Fields:
    mutateTime: When the change occurred.
    mutatedBy: The email address of the user making the change.
  """
    mutateTime = _messages.StringField(1)
    mutatedBy = _messages.StringField(2)