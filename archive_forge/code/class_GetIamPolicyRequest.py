from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetIamPolicyRequest(_messages.Message):
    """Request message for `GetIamPolicy` method.

  Fields:
    options: OPTIONAL: A `GetPolicyOptions` object for specifying options to
      `GetIamPolicy`.
  """
    options = _messages.MessageField('GetPolicyOptions', 1)