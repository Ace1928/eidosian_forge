from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def _GetAsset(parent, request_filter=None):
    asset_service_client = sc_client.AssetsClient()
    list_asset_response_for_project = asset_service_client.List(parent=parent, request_filter=request_filter)
    list_asset_results = list_asset_response_for_project.listAssetsResults
    if len(list_asset_results) != 1:
        raise InvalidSCCInputError('Something went wrong while retrieving the ProjectId for this Asset.')
    return list_asset_results[0].asset