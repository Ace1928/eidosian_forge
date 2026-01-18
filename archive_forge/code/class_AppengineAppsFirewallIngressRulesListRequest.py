from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsFirewallIngressRulesListRequest(_messages.Message):
    """A AppengineAppsFirewallIngressRulesListRequest object.

  Fields:
    matchingAddress: A valid IP Address. If set, only rules matching this
      address will be returned. The first returned rule will be the rule that
      fires on requests from this IP.
    pageSize: Maximum results to return per page.
    pageToken: Continuation token for fetching the next page of results.
    parent: Name of the Firewall collection to retrieve. Example:
      apps/myapp/firewall/ingressRules.
  """
    matchingAddress = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)