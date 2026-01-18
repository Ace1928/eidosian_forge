from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddStandaloneClusterMembershipResourceArg(parser, **kwargs):
    """Adds a resource argument for a bare metal standalone cluster membership.

  Args:
    parser: The argparse parser to add the resource arg to.
    **kwargs: Additional arguments like positional, required, etc.
  """
    positional = kwargs.get('positional')
    required = kwargs.get('required')
    name = 'membership' if positional else '--membership'
    concept_parsers.ConceptParser.ForResource(name, GetStandaloneClusterMembershipResourceSpec(), 'membership of the standalone cluster. Membership can be the membership ID or the full resource name.', required=required, flag_name_overrides={'project': '--membership-project', 'location': '--membership-location'}).AddToParser(parser)