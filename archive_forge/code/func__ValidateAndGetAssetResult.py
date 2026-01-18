from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def _ValidateAndGetAssetResult(response):
    list_asset_response = list(response)
    if not list_asset_response:
        raise InvalidSCCInputError('Asset or resource does not exist for the provided Organization. Please verify that both the OrganizationId and AssetId/ResourceName are correct.')
    if len(list_asset_response) > 1:
        raise InvalidSCCInputError('An asset can not have multiple projects. Something went wrong.')
    return list_asset_response[0]