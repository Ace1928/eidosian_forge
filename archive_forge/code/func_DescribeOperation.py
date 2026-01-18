from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def DescribeOperation(self, name):
    """Calls the Operations API."""
    req = self.messages.IdsProjectsLocationsOperationsGetRequest(name=name)
    return self._operations_client.Get(req)