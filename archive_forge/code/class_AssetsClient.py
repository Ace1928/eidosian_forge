from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
class AssetsClient(object):
    """Client for Security Center service in the for the Asset APIs."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClient()
        self.messages = messages or GetMessages()
        self._assetservice = self.client.organizations_assets

    def List(self, parent, request_filter=None):
        list_req_type = self.messages.SecuritycenterOrganizationsAssetsListRequest
        list_req = list_req_type(parent=parent, filter=request_filter)
        return self._assetservice.List(list_req)