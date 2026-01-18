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
def AddSecret(parser, purpose, positional=False, **kwargs):
    concept_parsers.ConceptParser.ForResource(name=_ArgOrFlag('secret', positional), resource_spec=GetSecretResourceSpec(), group_help='The secret {}.'.format(purpose), **kwargs).AddToParser(parser)