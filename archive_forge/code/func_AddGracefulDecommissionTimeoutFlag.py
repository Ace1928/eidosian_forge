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
def AddGracefulDecommissionTimeoutFlag(parser):
    """Adds a graceful decommission timeout for resizing a node group.

  Args:
    parser: The argparse parser for the command.
  """
    parser.add_argument('--graceful-decommission-timeout', help='Graceful decommission timeout for a node group scale-down resize.', required=False)