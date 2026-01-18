from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyFoldersPoliciesGetEffectivePolicyRequest(_messages.Message):
    """A OrgpolicyFoldersPoliciesGetEffectivePolicyRequest object.

  Fields:
    name: Required. The effective policy to compute. See Policy for naming
      requirements.
  """
    name = _messages.StringField(1, required=True)