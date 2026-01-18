from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.filestore import filestore_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.filestore import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddDisconnectManagedActiveDirectoryArg(parser):
    """Adds a --disconnect-managed-ad flag to the parser.

  Args:
    parser: argparse parser.
  """
    disconnnect_managed_ad_help = '        Disconnect the instance from Managed Active Directory.'
    parser.add_argument('--disconnect-managed-ad', action='store_true', required=False, help=disconnnect_managed_ad_help)