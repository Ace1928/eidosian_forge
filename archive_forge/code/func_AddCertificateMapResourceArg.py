from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCertificateMapResourceArg(parser, verb, name='map', noun=None, positional=True, required=True, with_location=True):
    """Add a resource argument for a Certificate Manager certificate map.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    name: str, the name of the main arg for the resource.
    noun: str, the resource; default: 'The certificate map'.
    positional: bool, if True, means that the map ID is a positional arg rather
      than a flag.
    required: bool, if False, means that map ID is optional.
    with_location: bool, if False, means that location flag is hidden.
  """
    flag_name_overrides = {}
    if not with_location:
        flag_name_overrides['location'] = ''
    noun = noun or 'The certificate map'
    concept_parsers.ConceptParser.ForResource(name if positional else '--' + name, GetCertificateMapResourceSpec(), '{} {}.'.format(noun, verb), required=required, flag_name_overrides=flag_name_overrides).AddToParser(parser)