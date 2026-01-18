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
def _GetConversionWorkspace(self, args):
    """Returns a conversion workspace."""
    conversion_workspace_type = self.messages.ConversionWorkspace
    if args.global_settings is None:
        args.global_settings = {}
    args.global_settings['filter'] = '*'
    args.global_settings['v2'] = 'true'
    if args.source_database_engine == 'ORACLE' and args.destination_database_engine == 'POSTGRESQL':
        args.global_settings['cc'] = 'true'
    global_settings = labels_util.ParseCreateArgs(args, conversion_workspace_type.GlobalSettingsValue, 'global_settings')
    source = self._GetDatabaseEngineInfo(args.source_database_engine, args.source_database_version)
    destination = self._GetDatabaseEngineInfo(args.destination_database_engine, args.destination_database_version)
    return conversion_workspace_type(globalSettings=global_settings, displayName=args.display_name, source=source, destination=destination)