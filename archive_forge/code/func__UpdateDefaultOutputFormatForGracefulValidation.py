from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import instance_groups_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.instance_groups.managed import flags as mig_flags
from the managed instance group, use the abandon-instances command instead.
def _UpdateDefaultOutputFormatForGracefulValidation(self, args):
    if args.IsSpecified('format'):
        return
    if args.skip_instances_on_validation_error:
        args.format = mig_flags.GetCommonPerInstanceCommandOutputFormat(with_validation_error=True)