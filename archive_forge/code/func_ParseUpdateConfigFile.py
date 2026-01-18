from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.media.asset import utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def ParseUpdateConfigFile(ref, args, req):
    """Reads the json file with asset type configs and update mask, then parse the cotent to the request message."""
    del ref
    update_file_config = json.loads(args.update_asset_type_config_file)
    messages = apis.GetMessagesModule('mediaasset', 'v1alpha')
    if 'assetType' not in update_file_config:
        raise exceptions.Error('assetType needs to be included in the config file.')
    if 'updateMask' not in update_file_config:
        raise exceptions.Error('updateMask needs to be included in the config file.')
    update_mask = update_file_config['updateMask']
    asset_type = update_file_config['assetType']
    if not isinstance(update_mask, list):
        raise exceptions.Error('updateMask needs to be a list.')
    if len(update_mask) != len(asset_type):
        raise exceptions.Error('updated assetType does not match with updateMask.')
    for update in update_mask:
        if update not in asset_type:
            raise exceptions.Error('updated assetType does not match with updateMask.')
    at = encoding.DictToMessage(asset_type, messages.AssetType)
    utils.ValidateMediaAssetMessage(at)
    req.assetType = at
    req.updateMask = ','.join(update_mask)
    return req