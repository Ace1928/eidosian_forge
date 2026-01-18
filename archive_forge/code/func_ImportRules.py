from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def ImportRules(self, name, args=None):
    """Import rules in a conversion workspace.

    Args:
      name: str, the reference of the conversion workspace to import rules in.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      Operation: the operation for importing rules in the conversion workspace
    """
    import_rules_req_type = self.messages.DatamigrationProjectsLocationsConversionWorkspacesMappingRulesImportRequest
    import_rules_req = import_rules_req_type(parent=name, importMappingRulesRequest=self._GetImportMappingRulesRequest(args))
    return self._mapping_rules_service.Import(import_rules_req)