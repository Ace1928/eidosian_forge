from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddMirroredInstancesArg(parser, is_for_update=False):
    """Adds args to specify mirrored instances."""
    if is_for_update:
        instances = parser.add_mutually_exclusive_group(help='      Update the mirrored instances of this packet mirroring.\n      ')
        instances.add_argument('--add-mirrored-instances', type=arg_parsers.ArgList(), metavar='INSTANCE', help='List of instances to add to the packet mirroring.')
        instances.add_argument('--remove-mirrored-instances', type=arg_parsers.ArgList(), metavar='INSTANCE', help='List of instances to remove from the packet mirroring.')
        instances.add_argument('--set-mirrored-instances', type=arg_parsers.ArgList(), metavar='INSTANCE', help='List of instances to be mirrored on the packet mirroring.')
        instances.add_argument('--clear-mirrored-instances', action='store_true', default=None, help='If specified, clear the existing instances from the packet mirroring.')
    else:
        parser.add_argument('--mirrored-instances', type=arg_parsers.ArgList(), metavar='INSTANCE', help='        List of instances to be mirrored.\n        You can provide this as the full or valid partial URL to the instance.\n        For example, the following are valid values:\n          * https://compute.googleapis.com/compute/v1/projects/myproject/\n            zones/us-central1-a/instances/instance-\n          * projects/myproject/zones/us-central1-a/instances/instance-1\n        ')