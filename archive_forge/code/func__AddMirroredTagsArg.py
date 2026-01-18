from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddMirroredTagsArg(parser, is_for_update=False):
    """Adds args to specify mirrored tags."""
    if is_for_update:
        tags = parser.add_mutually_exclusive_group(help='      Update the mirrored tags of this packet mirroring.\n\n      To read more about configuring network tags, read this guide:\n      https://cloud.google.com/vpc/docs/add-remove-network-tags\n\n      The virtual machines with the provided tags must live\n      in zones contained in the same region as this packet mirroring.\n      ')
        tags.add_argument('--add-mirrored-tags', type=arg_parsers.ArgList(), metavar='TAG', help='List of tags to add to the packet mirroring.')
        tags.add_argument('--remove-mirrored-tags', type=arg_parsers.ArgList(), metavar='TAG', help='List of tags to remove from the packet mirroring.')
        tags.add_argument('--set-mirrored-tags', type=arg_parsers.ArgList(), metavar='TAG', help='List of tags to be mirrored on the packet mirroring.')
        tags.add_argument('--clear-mirrored-tags', action='store_true', default=None, help='If specified, clear the existing tags from the packet mirroring.')
    else:
        parser.add_argument('--mirrored-tags', type=arg_parsers.ArgList(), metavar='TAG', help='        List of virtual machine instance tags to be mirrored.\n\n        To read more about configuring network tags, read this guide:\n        https://cloud.google.com/vpc/docs/add-remove-network-tags\n\n        The virtual machines with the provided tags must live\n        in zones contained in the same region as this packet mirroring.\n        ')