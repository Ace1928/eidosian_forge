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
def AddPdDiskType(parser):
    """Adds a --pd-disk-type flag to the given parser."""
    help_text = '  Type of the persistent directory.'
    parser.add_argument('--pd-disk-type', choices=['pd-standard', 'pd-balanced', 'pd-ssd'], default='pd-standard', help=help_text)