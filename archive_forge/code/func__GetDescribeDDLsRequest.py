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
def _GetDescribeDDLsRequest(self, conversion_workspace_ref, page_size, page_token, args):
    """Returns describe ddl conversion workspace request."""
    describe_ddl_req = self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest(commitId=args.commit_id, conversionWorkspace=conversion_workspace_ref, uncommitted=args.uncommitted, view=self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.ViewValueValuesEnum.DATABASE_ENTITY_VIEW_FULL, pageSize=page_size, pageToken=page_token)
    if args.IsKnownAndSpecified('tree_type'):
        describe_ddl_req.tree = self._GetTreeType(args.tree_type)
    else:
        describe_ddl_req.tree = self.messages.DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest.TreeValueValuesEnum.DRAFT_TREE
    if args.IsKnownAndSpecified('filter'):
        args.filter, server_filter = filter_rewrite.Rewriter().Rewrite(args.filter)
        describe_ddl_req.filter = server_filter
    return describe_ddl_req