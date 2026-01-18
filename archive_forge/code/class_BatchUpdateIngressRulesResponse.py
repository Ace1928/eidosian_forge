from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchUpdateIngressRulesResponse(_messages.Message):
    """Response message for Firewall.UpdateAllIngressRules.

  Fields:
    ingressRules: The full list of ingress FirewallRules for this application.
  """
    ingressRules = _messages.MessageField('FirewallRule', 1, repeated=True)