from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddSizeFlag(parser):
    """Adds the size field for resizing node groups.

  Args:
    parser: The argparse parser for the command.
  """
    parser.add_argument('--size', help='New size for a node group.', type=int, required=True)