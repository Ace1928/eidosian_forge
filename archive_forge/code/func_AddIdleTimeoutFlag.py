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
def AddIdleTimeoutFlag(parser, use_default=True):
    """Adds an --idle-timeout flag to the given parser."""
    help_text = "  How long (in seconds) to wait before automatically stopping an instance that\n  hasn't received any user traffic. A value of 0 indicates that this instance\n  should never time out due to idleness.\n  "
    parser.add_argument('--idle-timeout', default=7200 if use_default else None, type=int, help=help_text)