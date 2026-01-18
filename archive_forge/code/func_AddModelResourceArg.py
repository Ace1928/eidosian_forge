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
def AddModelResourceArg(parser, verb, prompt_func=region_util.PromptForRegion):
    """Add a resource argument for a Vertex AI model.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    prompt_func: function, the function to prompt for region from list of
      available regions which returns a string for the region selected. Default
      is region_util.PromptForRegion which contains three regions,
      'us-central1', 'europe-west4', and 'asia-east1'.
  """
    name = 'model'
    concept_parsers.ConceptParser.ForResource(name, GetModelResourceSpec(prompt_func=prompt_func), 'Model {}.'.format(verb), required=True).AddToParser(parser)