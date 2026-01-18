from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionValue(_messages.Message):
    """The action to take.

        Fields:
          storageClass: Target storage class. Required iff the type of the
            action is SetStorageClass.
          type: Type of the action. Currently, only Delete and SetStorageClass
            are supported.
        """
    storageClass = _messages.StringField(1)
    type = _messages.StringField(2)