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
def AddRequiredPlatformVersion(parser: parser_arguments.ArgumentInterceptor):
    """Adds flags to specify required platform version.

  Args:
    parser: The argparse parser to add the flag to.
  """
    parser.add_argument('--required-platform-version', type=str, help='Platform version required for upgrading a user cluster. If the current platform version is lower than the required version, the platform version will be updated to the required version. If it is not installed in the platform, download the required version bundle.')