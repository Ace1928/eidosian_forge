from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.command_lib.util import completers
def AddCloningImagesArgs(parser, sources_group):
    """Add args to support image cloning."""
    sources_group.add_argument('--source-image', help="      The name of an image to clone. May be used with\n      ``--source-image-project'' to clone an image in a different\n      project.\n      ")
    sources_group.add_argument('--source-image-family', help="      The family of the source image. This will cause the latest non-\n      deprecated image in the family to be used as the source image.\n      May be used with ``--source-image-project'' to refer to an image\n      family in a different project.\n      ")
    parser.add_argument('--source-image-project', help="      The project name of the source image. Must also specify either\n      ``--source-image'' or ``--source-image-family'' when using\n      this flag.\n      ")