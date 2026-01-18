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
def _GetConvertConversionWorkspaceRequest(self, args):
    """Returns convert conversion workspace request."""
    convert_req_obj = self.messages.ConvertConversionWorkspaceRequest(autoCommit=args.auto_commit)
    if args.IsKnownAndSpecified('filter'):
        args.filter, server_filter = filter_rewrite.Rewriter().Rewrite(args.filter)
        convert_req_obj.filter = server_filter
    return convert_req_obj