from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddConversionWorkspaceApplyResourceArg(parser, verb, positional=True):
    """Add a resource argument for applying a database migration cw.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to apply'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'conversion_workspace'
    else:
        name = '--conversion-workspace'
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetConversionWorkspaceResourceSpec(), 'The conversion workspace {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--destination-connection-profile', GetConnectionProfileResourceSpec(), 'The connection profile {} to.'.format(verb), flag_name_overrides={'region': ''}, required=True)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--destination-connection-profile.region': ['--region']}).AddToParser(parser)