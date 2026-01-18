from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MemberRestriction(_messages.Message):
    """The definition of MemberRestriction

  Fields:
    evaluation: The evaluated state of this restriction on a group.
    query: Member Restriction as defined by CEL expression. Supported
      restrictions are: `member.customer_id` and `member.type`. Valid values
      for `member.type` are `1`, `2` and `3`. They correspond to USER,
      SERVICE_ACCOUNT, and GROUP respectively. The value for
      `member.customer_id` only supports `groupCustomerId()` currently which
      means the customer id of the group will be used for restriction.
      Supported operators are `&&`, `||` and `==`, corresponding to AND, OR,
      and EQUAL. Examples: Allow only service accounts of given customer to be
      members. `member.type == 2 && member.customer_id == groupCustomerId()`
      Allow only users or groups to be members. `member.type == 1 ||
      member.type == 3`
  """
    evaluation = _messages.MessageField('RestrictionEvaluation', 1)
    query = _messages.StringField(2)