from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateAssetForUpdateRequest(args):
    """Create Asset for Message Update Requests."""
    module = dataplex_api.GetMessageModule()
    asset = module.GoogleCloudDataplexV1Asset(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1Asset, args), discoverySpec=GenerateDiscoverySpec(args))
    if args.IsSpecified('resource_read_access_mode'):
        setattr(asset, 'resourceSpec', module.GoogleCloudDataplexV1AssetResourceSpec(readAccessMode=module.GoogleCloudDataplexV1AssetResourceSpec.ReadAccessModeValueValuesEnum(args.resource_read_access_mode)))
    return asset