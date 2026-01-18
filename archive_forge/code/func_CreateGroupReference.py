from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.instance_groups.managed import wait_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
def CreateGroupReference(self, client, resources, args):
    return instance_groups_flags.MULTISCOPE_INSTANCE_GROUP_MANAGER_ARG.ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.ZONE, scope_lister=flags.GetDefaultScopeLister(client))