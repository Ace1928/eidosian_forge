from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def DescribeAssociation(self, name):
    """Calls the GetAssociation API."""
    get_request = self.messages.NetworksecurityProjectsLocationsFirewallEndpointAssociationsGetRequest(name=name)
    return self._association_client.Get(get_request)