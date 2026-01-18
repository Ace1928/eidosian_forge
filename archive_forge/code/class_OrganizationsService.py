from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages as messages
class OrganizationsService(base_api.BaseApiService):
    """Service class for the organizations resource."""
    _NAME = 'organizations'

    def __init__(self, client):
        super(OrgpolicyV2.OrganizationsService, self).__init__(client)
        self._upload_configs = {}