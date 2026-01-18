from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserMakeAdmin(_messages.Message):
    """JSON request template for setting/revoking admin status of a user in

  Directory API.

  Fields:
    status: Boolean indicating new admin status of the user
  """
    status = _messages.BooleanField(1)