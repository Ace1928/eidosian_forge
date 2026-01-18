from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListIngressRulesResponse(_messages.Message):
    """Response message for Firewall.ListIngressRules.

  Fields:
    ingressRules: The ingress FirewallRules for this application.
    nextPageToken: Continuation token for fetching the next page of results.
  """
    ingressRules = _messages.MessageField('FirewallRule', 1, repeated=True)
    nextPageToken = _messages.StringField(2)