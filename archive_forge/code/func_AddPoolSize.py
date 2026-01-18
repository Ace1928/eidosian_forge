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
def AddPoolSize(parser, use_default=True):
    """Adds a --pool-size flag to the given parser."""
    help_text = '  Number of instances to pool for faster Workstation starup.'
    parser.add_argument('--pool-size', default=0 if use_default else None, type=int, help=help_text)