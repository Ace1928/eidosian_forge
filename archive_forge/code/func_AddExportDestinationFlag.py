from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddExportDestinationFlag(parser):
    """Adds a --destination flag for a storage export command to a parser.

  Args:
    parser: argparse.ArgumentParser, the parser to which to add the flag
  """
    base.Argument('--destination', metavar='DESTINATION', required=True, help='      The path to an existing local directory or a Cloud Storage\n      bucket/directory into which to export files.\n      ').AddToParser(parser)