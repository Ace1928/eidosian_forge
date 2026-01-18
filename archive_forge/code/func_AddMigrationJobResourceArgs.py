from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddMigrationJobResourceArgs(parser, verb, required=False):
    """Add resource arguments for creating/updating a database migration job.

  Args:
    parser: argparse.ArgumentParser, the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
    required: boolean, whether source/dest resource args are required.
  """
    resource_specs = [presentation_specs.ResourcePresentationSpec('migration_job', GetMigrationJobResourceSpec(), 'The migration job {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--source', GetConnectionProfileResourceSpec(), 'ID of the source connection profile, representing the source database.', required=required, flag_name_overrides={'region': ''}), presentation_specs.ResourcePresentationSpec('--destination', GetConnectionProfileResourceSpec(), 'ID of the destination connection profile, representing the destination database.', required=required, flag_name_overrides={'region': ''})]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--source.region': ['--region'], '--destination.region': ['--region']}).AddToParser(parser)