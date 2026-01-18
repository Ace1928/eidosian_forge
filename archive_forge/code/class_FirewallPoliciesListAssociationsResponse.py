from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallPoliciesListAssociationsResponse(_messages.Message):
    """A FirewallPoliciesListAssociationsResponse object.

  Fields:
    associations: A list of associations.
    kind: [Output Only] Type of firewallPolicy associations. Always
      compute#FirewallPoliciesListAssociations for lists of firewallPolicy
      associations.
  """
    associations = _messages.MessageField('FirewallPolicyAssociation', 1, repeated=True)
    kind = _messages.StringField(2, default='compute#firewallPoliciesListAssociationsResponse')