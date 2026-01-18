from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.privateca import locations
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.kms import resource_args as kms_args
from googlecloudsdk.command_lib.privateca import completers as privateca_completers
from googlecloudsdk.command_lib.privateca import exceptions as privateca_exceptions
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddCertificateTemplatePositionalResourceArg(parser, verb):
    """Add a positional resource argument for a certificate template.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    arg_name = 'CERTIFICATE_TEMPLATE'
    concept_parsers.ConceptParser.ForResource(arg_name, CreateCertificateTemplateResourceSpec(arg_name), 'The template {}.'.format(verb), required=True).AddToParser(parser)