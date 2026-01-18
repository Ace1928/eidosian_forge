from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetDryRunFlag(help_override=None):
    help_txt = help_override or 'If true and command fails print the underlying command that was executed and its exit status.'
    return base.Argument('--dry-run', action='store_true', required=False, help=help_txt)