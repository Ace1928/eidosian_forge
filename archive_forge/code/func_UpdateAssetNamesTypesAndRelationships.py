from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def UpdateAssetNamesTypesAndRelationships(self, args, feed_name, update_masks):
    """Get Updated assetNames, assetTypes and relationshipTypes."""
    feed = self.service.Get(self.message_module.CloudassetFeedsGetRequest(name=feed_name))
    asset_names = repeated.ParsePrimitiveArgs(args, 'asset_names', lambda: feed.assetNames)
    if asset_names is not None:
        update_masks.append('asset_names')
    else:
        asset_names = []
    asset_types = repeated.ParsePrimitiveArgs(args, 'asset_types', lambda: feed.assetTypes)
    if asset_types is not None:
        update_masks.append('asset_types')
    else:
        asset_types = []
    relationship_types = repeated.ParsePrimitiveArgs(args, 'relationship_types', lambda: feed.relationshipTypes)
    if relationship_types is not None:
        update_masks.append('relationship_types')
    else:
        relationship_types = []
    return (asset_names, asset_types, relationship_types)