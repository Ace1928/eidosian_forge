from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddCmekResourceArgs(parser):
    """Add a resource argument for a connection profile cmek.

  Args:
    parser: the parser for the command.
  """
    concept_parsers.ConceptParser.ForResource('--cmek-key', GetCmekKeyResourceSpec(), 'Name of the CMEK (customer-managed encryption key) used for the connection profile. For example, projects/myProject/locations/us-central1/keyRings/myKeyRing/cryptoKeys/myKey.', flag_name_overrides={'region': ''}).AddToParser(parser)