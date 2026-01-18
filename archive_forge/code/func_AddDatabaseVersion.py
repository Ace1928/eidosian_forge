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
def AddDatabaseVersion(parser, alloydb_messages):
    """Adds Database Version flag.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    alloydb_messages: Message module.
  """
    parser.add_argument('--database-version', required=False, type=alloydb_messages.Cluster.DatabaseVersionValueValuesEnum, choices=[alloydb_messages.Cluster.DatabaseVersionValueValuesEnum.POSTGRES_14, alloydb_messages.Cluster.DatabaseVersionValueValuesEnum.POSTGRES_15], help='Database version of the cluster.')