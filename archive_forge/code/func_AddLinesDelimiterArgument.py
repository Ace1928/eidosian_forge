from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddLinesDelimiterArgument(parser):
    """Add the 'lines-terminated-by' argument to the parser."""
    parser.add_argument('--lines-terminated-by', help='Specifies the character that split line records. The value of this argument has to be a character in Hex ASCII Code. For example, "0A" represents a new line. This flag is only available for MySQL. If this flag is not provided, a new line character will be used as the default value.')