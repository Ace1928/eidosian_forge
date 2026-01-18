from typing import Generator
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.core import log
from googlecloudsdk.generated_clients.apis.securitycentermanagement.v1 import securitycentermanagement_v1_messages as messages
class EffectiveSHACustomModuleClient(object):
    """Client for SHA effective custom module interaction with the Security Center Management API."""

    def __init__(self):
        self._client = apis.GetClientInstance('securitycentermanagement', 'v1').projects_locations_effectiveSecurityHealthAnalyticsCustomModules

    def Get(self, name: str) -> messages.EffectiveSecurityHealthAnalyticsCustomModule:
        """Get a SHA effective custom module."""
        req = messages.SecuritycentermanagementProjectsLocationsEffectiveSecurityHealthAnalyticsCustomModulesGetRequest(name=name)
        return self._client.Get(req)

    def List(self, page_size: int, parent: str, limit: int) -> Generator[messages.EffectiveSecurityHealthAnalyticsCustomModule, None, messages.ListEffectiveSecurityHealthAnalyticsCustomModulesResponse]:
        """List the details of the resident and descendant SHA effective custom modules."""
        req = messages.SecuritycentermanagementProjectsLocationsEffectiveSecurityHealthAnalyticsCustomModulesListRequest(pageSize=page_size, parent=parent)
        return list_pager.YieldFromList(self._client, request=req, limit=limit, field='effectiveSecurityHealthAnalyticsCustomModules', batch_size=page_size, batch_size_attribute='pageSize')