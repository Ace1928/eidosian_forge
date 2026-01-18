from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import errno
import io
import os
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.util import files
import six
def GetAssetInventoryListInput(folder, project, org, file_path=None, asset_types_filter=None, filter_expression=None, krm_kind_filter=None):
    """Generate a AssetInventory export data set from api list call.


  Calls AssetInventory List API via shared api client (AssetListClient) and
  generates a list of exportable assets. If `asset_types_filter`,
  `gvk_kind_filter` or `filter_expression` is passed, it will filter out
  non-matching resources. If `file_path` is None list will be returned as a
  string otherwise it is written to disk at specified path.

  Args:
    folder: string, folder parent for resource export.
    project: string, project parent for resource export.
    org: string, organization parent for resource export.
    file_path: string, path to write AssetInventory export file to. If None,
      results are returned as string.
    asset_types_filter: [string], list of asset types to include in the output
      file.
    filter_expression: string, a valid gcloud filter expression. See `gcloud
      topic filter` for more details.
    krm_kind_filter: [string], list of KrmKinds corresponding to asset types to
      include in the output.

  Returns:
    string: file path where AssetInventory data has been written or raw data if
      `temp_file_path` is None. Returns None if no results returned from API.

  Raises:
    RequiredArgumentException: If none of folder, project or org is provided.
    ResourceNotFoundException: If no resources are found or returned from
      filtering.
    ClientException: Writing file to disk.
  """
    root_asset = asset_utils.GetParentNameForExport(organization=org, project=project, folder=folder)
    asset_client = client_util.AssetListClient(root_asset)
    filter_func = resource_filter.Compile(filter_expression.strip()).Evaluate if filter_expression else None
    asset_filter = asset_types_filter or []
    if krm_kind_filter:
        kind_filters = _BuildAssetTypeFilterFromKind(krm_kind_filter)
        if not kind_filters:
            raise ResourceNotFoundException('No matching resource types found for {}'.format(krm_kind_filter))
        asset_filter.extend(kind_filters)
    args = ApiClientArgs(snapshot_time=None, limit=None, page_size=None, content_type=None, asset_types=sorted(asset_filter), parent=root_asset, filter_func=filter_func, relationship_types=[])
    asset_results = asset_client.List(args, do_filter=True)
    asset_string_array = []
    for item in asset_results:
        item_str = encoding.MessageToJson(item)
        item_str = item_str.replace('"assetType"', '"asset_type"')
        asset_string_array.append(item_str)
    if not asset_string_array:
        if asset_types_filter:
            asset_msg = '\n With resource types in [{}].'.format(asset_types_filter)
        else:
            asset_msg = ''
        if filter_expression:
            filter_msg = '\n Matching provided filter [{}].'.format(filter_expression)
        else:
            filter_msg = ''
        raise ResourceNotFoundException('No matching resources found for [{parent}] {assets} {filter}'.format(parent=root_asset, assets=asset_msg, filter=filter_msg))
    if not file_path:
        return '\n'.join(asset_string_array)
    else:
        try:
            files.WriteFileAtomically(file_path, '\n'.join(asset_string_array))
        except (ValueError, TypeError) as e:
            raise ClientException(e)
        return file_path