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
def AddImportDestinationFlag(parser, folder):
    """Adds a --destination flag for a storage import command to a parser.

  Args:
    parser: argparse.ArgumentParser, the parser to which to add the flag
    folder: str, the top-level folder in the bucket into which the import
      command will write. Should not contain any slashes. For example, 'dags'.
  """
    base.Argument('--destination', metavar='DESTINATION', required=False, help="      An optional subdirectory under the {}/ directory in the environment's\n      Cloud Storage bucket into which to import files. May contain forward\n      slashes to delimit multiple levels of subdirectory nesting, but should not\n      contain leading or trailing slashes. If the DESTINATION does not exist, it\n      will be created.\n      ".format(folder)).AddToParser(parser)