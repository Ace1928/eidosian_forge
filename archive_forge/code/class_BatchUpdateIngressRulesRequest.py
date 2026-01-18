from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchUpdateIngressRulesRequest(_messages.Message):
    """Request message for Firewall.BatchUpdateIngressRules.

  Fields:
    ingressRules: A list of FirewallRules to replace the existing set.
  """
    ingressRules = _messages.MessageField('FirewallRule', 1, repeated=True)