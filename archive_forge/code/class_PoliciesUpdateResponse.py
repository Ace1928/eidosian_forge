from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PoliciesUpdateResponse(_messages.Message):
    """A PoliciesUpdateResponse object.

  Fields:
    header: A ResponseHeader attribute.
    policy: A Policy attribute.
  """
    header = _messages.MessageField('ResponseHeader', 1)
    policy = _messages.MessageField('Policy', 2)