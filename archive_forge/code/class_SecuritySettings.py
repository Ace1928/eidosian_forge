from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritySettings(_messages.Message):
    """The definition of security settings.

  Fields:
    memberRestriction: The Member Restriction value
    name: Output only. The resource name of the security settings. Shall be of
      the form `groups/{group_id}/securitySettings`.
  """
    memberRestriction = _messages.MessageField('MemberRestriction', 1)
    name = _messages.StringField(2)