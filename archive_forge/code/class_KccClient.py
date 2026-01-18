from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.util.declarative.clients import declarative_client_base
from googlecloudsdk.command_lib.util.resource_map.declarative import resource_name_translator
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
class KccClient(declarative_client_base.DeclarativeClientBase):
    """Binary Client Interface for the config-connector binary tool."""

    @property
    def binary_name(self):
        return 'config-connector'

    @property
    def binary_prompt(self):
        return 'This command requires the `config-connector` binary to be installed to export GCP resource configurations. Would you like to install the `config-connector` binary to continue command execution?'

    def _GetBinarySpecificExportArguments(self, args, cmd):
        return cmd

    def BulkExport(self, args):
        CheckForAssetInventoryEnablementWithPrompt(getattr(args, 'project', None))
        if args.IsSpecified('resource_types') or args.IsSpecified('resource_types_file'):
            return self._CallBulkExportFromAssetList(args)
        cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_filter=True)
        return self._CallBulkExport(cmd, args, asset_list_input=None)

    def _ParseKindTypesFileData(self, file_data):
        """Parse Resource Types data into input string Array."""
        if not file_data:
            return None
        return [x for x in re.split('\\s+|,+', file_data) if x]

    def _CallBulkExportFromAssetList(self, args):
        """BulkExport with support for resource kind/asset type and filtering."""
        CheckForAssetInventoryEnablementWithPrompt(getattr(args, 'project', None))
        kind_args = self._ParseResourceTypes(args)
        asset_list_input = declarative_client_base.GetAssetInventoryListInput(folder=getattr(args, 'folder', None), project=getattr(args, 'project', None), org=getattr(args, 'organization', None), krm_kind_filter=kind_args, filter_expression=getattr(args, 'filter', None))
        cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
        return self._CallBulkExport(cmd, args, asset_list_input=asset_list_input)

    def ExportAll(self, args, collection):
        """Exports all resources of a particular collection."""
        cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
        asset_type = [_TranslateCollectionToAssetType(collection)]
        asset_list_input = declarative_client_base.GetAssetInventoryListInput(folder=getattr(args, 'folder', None), project=getattr(args, 'project', None) or properties.VALUES.core.project.GetOrFail(), org=getattr(args, 'organization', None), asset_types_filter=asset_type, filter_expression=getattr(args, 'filter', None))
        cmd = self._GetBinaryExportCommand(args, 'bulk-export', skip_parent=True, skip_filter=True)
        return self._CallBulkExport(cmd, args, asset_list_input)