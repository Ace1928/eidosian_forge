from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def _CallBulkExportFromAssetList(self, args):
    """BulkExport with support for resource kind/asset type and filtering."""
    CheckForAssetInventoryEnablementWithPrompt(getattr(args, 'project', None))
    kind_args = self._ParseResourceTypes(args)
    asset_list_input = declarative_client_base.GetAssetInventoryListInput(folder=getattr(args, 'folder', None), project=getattr(args, 'project', None), org=getattr(args, 'organization', None), krm_kind_filter=kind_args, filter_expression=getattr(args, 'filter', None))
    cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
    return self._CallBulkExport(cmd, args, asset_list_input=asset_list_input)