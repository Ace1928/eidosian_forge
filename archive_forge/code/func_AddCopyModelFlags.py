from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddCopyModelFlags(parser, prompt_func=region_util.PromptForRegion):
    """Adds flags for AddCopyModelFlags.

  Args:
    parser: the parser for the command.
    prompt_func: function, the function to prompt a list of available regions
      and return a string of the region that is selected by user.
  """
    AddRegionResourceArg(parser, 'to copy the model into', prompt_func=prompt_func)
    base.Argument('--source-model', required=True, help='The resource name of the Model to copy. That Model must be in the same Project.\nFormat: `projects/{project}/locations/{location}/models/{model}`.\n').AddToParser(parser)
    base.Argument('--kms-key-name', help='The Cloud KMS resource identifier of the customer managed encryption key\nused to protect the resource.\nHas the form:\n`projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-key`.\nThe key needs to be in the same region as the destination region of the model to be copied.\n').AddToParser(parser)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--destination-model-id', type=str, help='Copy source_model into a new Model with this ID. The ID will become the final component of the model resource name.\nThis value may be up to 63 characters, and valid characters are `[a-z0-9_-]`. The first character cannot be a number or hyphen.\n')
    group.add_argument('--destination-parent-model', type=str, help='Specify this field to copy source_model into this existing Model as a new version.\nFormat: `projects/{project}/locations/{location}/models/{model}`.\n')