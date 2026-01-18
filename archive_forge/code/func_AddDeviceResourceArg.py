from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddDeviceResourceArg(parser, verb, positional=True):
    """Add a resource argument for a cloud IOT device.

  NOTE: Must be used only if it's the only resource arg in the command.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the device ID is a positional rather
      than a flag.
  """
    if positional:
        name = 'device'
    else:
        name = '--device'
    concept_parsers.ConceptParser.ForResource(name, GetDeviceResourceSpec(), 'The device {}.'.format(verb), required=True).AddToParser(parser)