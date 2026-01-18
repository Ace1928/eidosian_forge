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
def AddDriverPoolId(parser):
    """Adds the customer provided driver pool id field.

  Args:
    parser: The argparse parser for the command.
  """
    parser.add_argument('--driver-pool-id', help='\n            Custom identifier for the DRIVER Node Group being created. If not\n            provided, a random string is generated.\n            ', required=False, default=None)