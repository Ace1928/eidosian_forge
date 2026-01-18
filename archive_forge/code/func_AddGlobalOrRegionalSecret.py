from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.command_lib.secrets import completers as secrets_completers
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddGlobalOrRegionalSecret(parser, purpose='create a secret', **kwargs):
    """Adds a secret resource.

  Secret resource can be global secret or regional secret. If command has
  "--location" then regional secret will be created or else global secret will
  be created.
  Regionl secret - projects/<project>/locations/<location>/secrets/<secret>
  Global secret - projects/<project>/secrets/<secret>

  Args:
      parser: given argument parser
      purpose: help text
      **kwargs: extra arguments
  """
    secret_or_region_secret_spec = multitype.MultitypeResourceSpec('global or regional secret', GetSecretResourceSpec(), GetRegionalSecretResourceSpec(), allow_inactive=True, **kwargs)
    concept_parsers.ConceptParser([presentation_specs.MultitypeResourcePresentationSpec('secret', secret_or_region_secret_spec, purpose, required=True, hidden=True)]).AddToParser(parser)