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
def AddEnableSSHToVM(parser):
    """Adds a --enable-ssh-to-vm flag to the given parser."""
    help_text = '  If set, workstations disable SSH connections to the root VM.'
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--disable-ssh-to-vm', action='store_true', help=help_text)
    help_text = '  If set, workstations enable SSH connections to the root VM.'
    group.add_argument('--enable-ssh-to-vm', action='store_true', help=help_text)