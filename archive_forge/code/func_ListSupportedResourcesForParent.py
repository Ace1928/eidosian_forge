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
def ListSupportedResourcesForParent(self, project=None, organization=None, folder=None):
    """List all exportable resource types for a given project, org or folder."""
    if not (project or organization or folder):
        raise ClientException('At least one of project, organization or folder must be specified for this operation')
    name_translator = resource_name_translator.ResourceNameTranslator()
    asset_list_data = GetAssetInventoryListInput(folder=folder, org=organization, project=project)
    asset_types = set([x.replace('"', '') for x in _ASSET_TYPE_REGEX.findall(asset_list_data)])
    exportable_kinds = []
    for asset in asset_types:
        try:
            meta_resource = name_translator.get_resource(asset_inventory_type=asset)
            gvk = KrmGroupValueKind(kind=meta_resource.krm_kind.krm_kind, group=meta_resource.krm_kind.krm_group + _KRM_GROUP_SUFFIX, bulk_export_supported=meta_resource.resource_data.support_bulk_export, export_supported=meta_resource.resource_data.support_single_export, iam_supported=meta_resource.resource_data.support_iam)
            exportable_kinds.append(gvk)
        except resource_name_translator.ResourceIdentifierNotFoundError:
            continue
    return sorted(exportable_kinds, key=lambda x: x.kind)