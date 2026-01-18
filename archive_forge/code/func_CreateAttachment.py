from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CreateAttachment(self, attachment_id, parent, producer_forwarding_rule_name, labels=None):
    """Calls the CreateAttachment API."""
    attachment = self.messages.FirewallAttachment(producerForwardingRuleName=producer_forwarding_rule_name, labels=labels)
    create_request = self.messages.NetworksecurityProjectsLocationsFirewallAttachmentsCreateRequest(firewallAttachment=attachment, firewallAttachmentId=attachment_id, parent=parent)
    return self._attachment_client.Create(create_request)