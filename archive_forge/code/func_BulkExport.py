from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def BulkExport(self, args):
    CheckForAssetInventoryEnablementWithPrompt(getattr(args, 'project', None))
    if args.IsSpecified('resource_types') or args.IsSpecified('resource_types_file'):
        return self._CallBulkExportFromAssetList(args)
    cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_filter=True)
    return self._CallBulkExport(cmd, args, asset_list_input=None)