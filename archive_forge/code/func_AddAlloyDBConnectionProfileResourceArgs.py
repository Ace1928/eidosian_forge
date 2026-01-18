from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAlloyDBConnectionProfileResourceArgs(parser, verb):
    """Add resource arguments for a database migration AlloyDB connection profile.

  Args:
    parser: the parser for the command.
    verb: str, the verb to describe the resource, such as 'to update'.
  """
    resource_specs = [presentation_specs.ResourcePresentationSpec('connection_profile', GetConnectionProfileResourceSpec(), 'The connection profile {}.'.format(verb), required=True), presentation_specs.ResourcePresentationSpec('--kms-key', GetKMSKeyResourceSpec(), 'Name of the CMEK (customer-managed encryption key) used for this AlloyDB cluster. For example, projects/myProject/locations/us-central1/keyRings/myKeyRing/cryptoKeys/myKey.', flag_name_overrides={'region': ''})]
    concept_parsers.ConceptParser(resource_specs, command_level_fallthroughs={'--kms-key.region': ['--region']}).AddToParser(parser)