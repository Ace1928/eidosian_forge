from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.resource_policies import util as maintenance_util
from googlecloudsdk.core.util import times
import six
def ResolveSourceInstanceTemplate(args, resources):
    return compute_flags.ResourceArgument('--source-instance-template', resource_name='instance template', scope_flags_usage=compute_flags.ScopeFlagsUsage.DONT_USE_SCOPE_FLAGS, global_collection='compute.instanceTemplates', regional_collection='compute.regionInstanceTemplates').ResolveAsResource(args, resources, default_scope=compute_scope.ScopeEnum.GLOBAL)