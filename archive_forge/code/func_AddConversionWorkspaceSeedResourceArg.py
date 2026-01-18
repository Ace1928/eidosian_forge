from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddConversionWorkspaceSeedResourceArg(parser, verb, positional=True):
    """Add a resource argument for seeding a database migration cw.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to seed'.
    positional: bool, if True, means that the resource is a positional rather
      than a flag.
  """
    if positional:
        name = 'conversion_workspace'
    else:
        name = '--conversion-workspace'
    connection_profile = parser.add_group(mutex=True, required=True)
    resource_specs = [presentation_specs.ResourcePresentationSpec(name, GetConversionWorkspaceResourceSpec(), 'The conversion workspace {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--source-connection-profile', GetConnectionProfileResourceSpec(), 'The connection profile {} from.'.format(verb), flag_name_overrides={'region': ''}, group=connection_profile), presentation_specs.ResourcePresentationSpec('--destination-connection-profile', GetConnectionProfileResourceSpec(), 'The connection profile {} from.'.format(verb), flag_name_overrides={'region': ''}, group=connection_profile)]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--source-connection-profile.region': ['--region'], '--destination-connection-profile.region': ['--region']}).AddToParser(parser)