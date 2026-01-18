from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def ListAttachments(self, parent, limit=None, page_size=None):
    """Calls the ListAttachments API."""
    list_request = self.messages.NetworksecurityProjectsLocationsFirewallAttachmentsListRequest(parent=parent)
    return list_pager.YieldFromList(self._attachment_client, list_request, batch_size=page_size, limit=limit, field='firewallAttachments', batch_size_attribute='pageSize')