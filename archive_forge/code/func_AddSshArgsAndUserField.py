from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddSshArgsAndUserField(parser):
    """Adds a --user flag to the given parser."""
    help_text = '  The username with which to SSH.\n  '
    parser.add_argument('--user', type=str, default='user', help=help_text)
    help_text = '  Flags and positionals passed to the underlying ssh implementation.'
    parser.add_argument('ssh_args', nargs=argparse.REMAINDER, help=help_text)