from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def ExportAll(self, args, collection):
    """Exports all resources of a particular collection."""
    cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
    asset_type = [_TranslateCollectionToAssetType(collection)]
    asset_list_input = declarative_client_base.GetAssetInventoryListInput(folder=getattr(args, 'folder', None), project=getattr(args, 'project', None) or properties.VALUES.core.project.GetOrFail(), org=getattr(args, 'organization', None), asset_types_filter=asset_type, filter_expression=getattr(args, 'filter', None))
    cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
    return self._CallBulkExport(cmd, args, asset_list_input)