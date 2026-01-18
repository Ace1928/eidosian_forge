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
def _GetSeedConversionWorkspaceRequest(self, source_connection_profile_ref, destination_connection_profile_ref, args):
    """Returns seed conversion workspace request."""
    seed_cw_request = self.messages.SeedConversionWorkspaceRequest(autoCommit=args.auto_commit)
    if source_connection_profile_ref is not None:
        seed_cw_request.sourceConnectionProfile = source_connection_profile_ref.RelativeName()
    if destination_connection_profile_ref is not None:
        seed_cw_request.destinationConnectionProfile = destination_connection_profile_ref.RelativeName()
    return seed_cw_request