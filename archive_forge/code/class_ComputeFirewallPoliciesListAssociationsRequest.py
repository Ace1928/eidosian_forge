from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeFirewallPoliciesListAssociationsRequest(_messages.Message):
    """A ComputeFirewallPoliciesListAssociationsRequest object.

  Fields:
    targetResource: The target resource to list associations. It is an
      organization, or a folder.
  """
    targetResource = _messages.StringField(1)