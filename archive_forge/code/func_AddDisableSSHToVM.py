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
def AddDisableSSHToVM(parser, use_default=True):
    """Adds a --disable-ssh-to-vm flag to the given parser."""
    help_text = '  Default value is False.\n  If set, workstations disable SSH connections to the root VM.'
    parser.add_argument('--disable-ssh-to-vm', action='store_true', default=False if use_default else False, help=help_text)