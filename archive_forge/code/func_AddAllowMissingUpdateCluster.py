from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddAllowMissingUpdateCluster(parser: parser_arguments.ArgumentInterceptor):
    """Adds a flag to enable allow missing in an update cluster request.

  If set to true, and the cluster is not found, the request will
  create a new cluster with the provided configuration. The user
  must have both create and update permission to call Update with
  allow_missing set to true.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--allow-missing', action='store_true', hidden=True, help='If set, and the Anthos cluster on VMware is not found, the update request will try to create a new cluster with the provided configuration.')