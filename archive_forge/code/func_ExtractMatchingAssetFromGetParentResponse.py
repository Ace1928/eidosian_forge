from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py.extra_types import _JsonValueToPythonValue
from googlecloudsdk.api_lib.scc import securitycenter_client as sc_client
from googlecloudsdk.command_lib.scc.errors import InvalidSCCInputError
def ExtractMatchingAssetFromGetParentResponse(response, args):
    """Returns Parent for the user provided asset or resource-name."""
    del args
    asset_result = _ValidateAndGetAssetResult(response)
    asset_parent = _GetAssetResourceParent(asset_result)
    organization = _ExtractOrganization(asset_result)
    resource_name_filter = _FilterOnResourceName(asset_parent)
    asset = _GetAsset(organization, resource_name_filter)
    parent = _GetParent(asset)
    result_dictionary = {'parent': parent}
    return result_dictionary