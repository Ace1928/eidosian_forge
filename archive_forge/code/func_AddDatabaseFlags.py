from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddDatabaseFlags(parser, update=False):
    """Adds a `--database-flags` flag to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    update: Whether or not database flags were provided as part of an update.
  """
    help_ = 'Comma-separated list of database flags to set on the instance. Use an equals sign to separate flag name and value. Flags without values, like skip_grant_tables, can be written out without a value after, e.g., `skip_grant_tables=`. Use on/off for booleans. View the Instance Resource API for allowed flags. (e.g., `--database-flags max_allowed_packet=55555,skip_grant_tables=,log_output=1`)'
    if update:
        help_ += '\n\nThe value given for this argument *replaces* the existing list.'
    parser.add_argument('--database-flags', type=arg_parsers.ArgDict(min_length=1), metavar='FLAG=VALUE', required=False, help=help_)