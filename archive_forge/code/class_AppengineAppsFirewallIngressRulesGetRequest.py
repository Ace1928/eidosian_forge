from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppengineAppsFirewallIngressRulesGetRequest(_messages.Message):
    """A AppengineAppsFirewallIngressRulesGetRequest object.

  Fields:
    name: Name of the Firewall resource to retrieve. Example:
      apps/myapp/firewall/ingressRules/100.
  """
    name = _messages.StringField(1, required=True)