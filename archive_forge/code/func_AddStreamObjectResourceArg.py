from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddStreamObjectResourceArg(parser):
    """Add a resource argument for a Datastream stream object.

  Args:
    parser: the parser for the command.
  """
    resource_specs = [presentation_specs.ResourcePresentationSpec('--stream', GetStreamResourceSpec(), 'The stream to list objects for.', required=True)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--stream.location': ['--location']}).AddToParser(parser)