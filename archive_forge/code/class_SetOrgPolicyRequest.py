from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetOrgPolicyRequest(_messages.Message):
    """The request sent to the SetOrgPolicyRequest method.

  Fields:
    policy: `Policy` to set on the resource.
  """
    policy = _messages.MessageField('OrgPolicy', 1)