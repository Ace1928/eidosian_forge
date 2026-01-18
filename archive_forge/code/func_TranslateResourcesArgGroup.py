from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
from googlecloudsdk.command_lib.compute.reservations import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
def TranslateResourcesArgGroup(messages, args):
    """Util functions to parse ResourceCommitments."""
    resources_arg = args.resources
    resources = TranslateResourcesArg(messages, resources_arg)
    if 'local-ssd' in resources_arg.keys():
        resources.append(messages.ResourceCommitment(amount=resources_arg['local-ssd'], type=messages.ResourceCommitment.TypeValueValuesEnum.LOCAL_SSD))
    if args.IsSpecified('resources_accelerator'):
        accelerator_arg = args.resources_accelerator
        resources.append(messages.ResourceCommitment(amount=accelerator_arg['count'], acceleratorType=accelerator_arg['type'], type=messages.ResourceCommitment.TypeValueValuesEnum.ACCELERATOR))
    return resources