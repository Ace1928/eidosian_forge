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
def AddGlobalOrRegionalVersionOrAlias(parser, purpose='create a version alias', **kwargs):
    """Adds a version resource or alias.

  Args:
      parser: given argument parser
      purpose: help text
      **kwargs: extra arguments
  """
    global_or_region_version_spec = multitype.MultitypeResourceSpec('global or regional secret version', GetVersionResourceSpec(), GetRegionalVersionResourceSpec(), allow_inactive=True, **kwargs)
    concept_parsers.ConceptParser([presentation_specs.MultitypeResourcePresentationSpec('version', global_or_region_version_spec, purpose, required=True, hidden=True)]).AddToParser(parser)