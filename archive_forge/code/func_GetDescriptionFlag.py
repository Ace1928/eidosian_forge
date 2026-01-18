from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetDescriptionFlag():
    """Returns the flag for SSL policy description."""
    return base.Argument('--description', help='An optional, textual description for the SSL policy.')