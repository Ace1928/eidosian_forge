from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
def AddAdminConfigType(parser: parser_arguments.ArgumentInterceptor):
    """Adds flags to specify admin cluster version config type.

  Args:
    parser: The argparse parser to add the flag to.
  """
    config_type_group = parser.add_group('Use cases for querying versions.', mutex=True)
    upgrade_config = config_type_group.add_group('Upgrade an Anthos on bare metal user cluster use case.')
    arg_parser = concept_parsers.ConceptParser([presentation_specs.ResourcePresentationSpec('--admin-cluster', GetAdminClusterResourceSpec(), 'Admin cluster to query versions for upgrade.', flag_name_overrides={'location': ''}, required=False, group=upgrade_config)], command_level_fallthroughs={'--admin-cluster.location': ['--location']})
    arg_parser.AddToParser(parser)