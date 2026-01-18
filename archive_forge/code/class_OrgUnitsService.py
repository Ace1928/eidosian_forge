from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudidentity.v1beta1 import cloudidentity_v1beta1_messages as messages
class OrgUnitsService(base_api.BaseApiService):
    """Service class for the orgUnits resource."""
    _NAME = 'orgUnits'

    def __init__(self, client):
        super(CloudidentityV1beta1.OrgUnitsService, self).__init__(client)
        self._upload_configs = {}