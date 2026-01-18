from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _ValidateConversionWorkspaceArgs(self, conversion_workspace_ref, args):
    """Validate flags for conversion workspace.

    Args:
      conversion_workspace_ref: str, the reference of the conversion workspace.
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Raises:
      BadArgumentException: commit-id or filter field is provided without
      specifying the conversion workspace
    """
    if conversion_workspace_ref is None:
        if args.IsKnownAndSpecified('commit_id'):
            raise exceptions.BadArgumentException('commit-id', 'Conversion workspace commit-id can only be specified for migration jobs associated with a conversion workspace.')
        if args.IsKnownAndSpecified('filter'):
            raise exceptions.BadArgumentException('filter', 'Filter can only be specified for migration jobs associated with a conversion workspace.')