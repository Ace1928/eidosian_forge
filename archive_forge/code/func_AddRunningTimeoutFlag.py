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
def AddRunningTimeoutFlag(parser, use_default=True):
    """Adds an --running-timeout flag to the given parser."""
    help_text = '  How long (in seconds) to wait before automatically stopping a workstation\n  after it started. A value of 0 indicates that workstations using this config\n  should never time out.\n  '
    parser.add_argument('--running-timeout', default=7200 if use_default else None, type=int, help=help_text)