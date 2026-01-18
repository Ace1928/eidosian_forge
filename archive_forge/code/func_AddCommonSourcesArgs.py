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
def AddCommonSourcesArgs(parser, sources_group):
    """Add common args for specifying the source for image creation."""
    sources_group.add_argument('--source-uri', help="      The full Cloud Storage URI where the disk image is stored.\n      This file must be a gzip-compressed tarball whose name ends in\n      ``.tar.gz''.\n      For more information about Cloud Storage URIs,\n      see https://cloud.google.com/storage/docs/request-endpoints#json-api.\n      ")
    SOURCE_DISK_ARG.AddArgument(parser, mutex_group=sources_group)