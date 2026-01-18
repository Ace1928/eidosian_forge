from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.container.bare_metal import cluster_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _AddAuthorization(bare_metal_security_config_group, is_update=False):
    """Adds flags to specify applied and managed RBAC policy.

  Args:
    bare_metal_security_config_group: The parent group to add the flags to.
    is_update: bool, whether the flag is for update command or not.
  """
    required = not is_update
    authorization_group = bare_metal_security_config_group.add_group(help='Cluster authorization configurations to bootstrap onto the standalone cluster')
    authorization_group.add_argument('--admin-users', help='Users that will be granted the cluster-admin role on the cluster, providing full access to the cluster.', action='append', required=required)