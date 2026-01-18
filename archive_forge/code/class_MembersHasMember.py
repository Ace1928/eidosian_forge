from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembersHasMember(_messages.Message):
    """JSON template for Has Member response in Directory API.

  Fields:
    isMember: Identifies whether the given user is a member of the group.
      Membership can be direct or nested.
  """
    isMember = _messages.BooleanField(1)