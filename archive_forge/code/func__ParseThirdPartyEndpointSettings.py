from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def _ParseThirdPartyEndpointSettings(self, target_firewall_attachment):
    if target_firewall_attachment is None:
        return None
    return self.messages.ThirdPartyEndpointSettings(targetFirewallAttachment=target_firewall_attachment)