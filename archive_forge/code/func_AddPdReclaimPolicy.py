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
def AddPdReclaimPolicy(parser):
    """Adds a --pd-reclaim-policy flag to the given parser."""
    help_text = '  What should happen to the disk after the Workstation is deleted.'
    parser.add_argument('--pd-reclaim-policy', choices={'delete': 'The persistent disk will be deleted with the Workstation.', 'retain': 'The persistent disk will be remain after the workstation is deleted and the administrator must manually delete the disk.'}, default='delete', help=help_text)