from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def ParseCreateConfigFile(ref, args, req):
    """Reads the json file of with the asset type configs and parse the content to the request message."""
    del ref
    messages = apis.GetMessagesModule('mediaasset', 'v1alpha')
    message_class = messages.AssetType
    if args.create_asset_type_config_file:
        asset_type_configs = json.loads(args.create_asset_type_config_file)
        at = encoding.DictToMessage(asset_type_configs, message_class)
        utils.ValidateMediaAssetMessage(at)
        req.assetType = at
    if args.IsKnownAndSpecified('labels'):
        req.assetType.labels = encoding.DictToMessage(args.labels, messages.AssetType.LabelsValue)
    return req