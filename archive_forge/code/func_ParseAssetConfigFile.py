from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import resources
def ParseAssetConfigFile(ref, args, req):
    """Prepare the asset for create and update requests."""
    del ref
    messages = apis.GetMessagesModule('mediaasset', 'v1alpha')
    if args.IsKnownAndSpecified('asset_config_file'):
        asset_data = json.loads(args.asset_config_file)
        asset = encoding.DictToMessage(asset_data, messages.Asset)
        utils.ValidateMediaAssetMessage(asset)
        req.asset = asset
    if args.IsKnownAndSpecified('labels'):
        req.asset.labels = encoding.DictToMessage(args.labels, messages.Asset.LabelsValue)
    return req