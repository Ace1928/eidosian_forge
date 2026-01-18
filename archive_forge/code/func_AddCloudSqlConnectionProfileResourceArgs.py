from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCloudSqlConnectionProfileResourceArgs(parser, verb):
    """Add resource arguments for a database migration CloudSQL connection profile.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    resource_specs = [presentation_specs.ResourcePresentationSpec('connection_profile', GetConnectionProfileResourceSpec(), 'The connection profile {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--source-id', GetConnectionProfileResourceSpec(), 'Database Migration Service source connection profile ID.', required=True, flag_name_overrides={'region': ''})]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--source-id.region': ['--region']}).AddToParser(parser)