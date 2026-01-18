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
def _GetUpdatedConversionWorkspace(self, conversion_workspace, args):
    """Returns updated conversion workspace and list of updated fields."""
    update_fields = self._GetUpdateMask(args)
    if args.IsSpecified('display_name'):
        conversion_workspace.displayName = args.display_name
    return (conversion_workspace, update_fields)