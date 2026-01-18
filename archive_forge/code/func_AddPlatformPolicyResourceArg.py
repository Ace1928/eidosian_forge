from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
from googlecloudsdk.command_lib.kms import flags as kms_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs as presentation_specs_lib
def AddPlatformPolicyResourceArg(parser, verb):
    """Add a resource argument for a policy.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    concept_parsers.ConceptParser.ForResource('policy_resource_name', _GetPlatformPolicyResourceSpec(), 'The resource name of the policy {}.'.format(verb), required=True).AddToParser(parser)