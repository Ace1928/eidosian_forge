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
def AddShieldedVtpm(parser, use_default=True):
    """Adds a --shielded-vtpm flag to the given parser."""
    help_text = '  Default value is false.\n  If set, instances will have vTPM enabled.'
    parser.add_argument('--shielded-vtpm', action='store_true', default=False if use_default else None, help=help_text)