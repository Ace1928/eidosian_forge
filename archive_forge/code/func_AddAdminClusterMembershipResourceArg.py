from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.gkeonprem import flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddAdminClusterMembershipResourceArg(parser: parser_arguments.ArgumentInterceptor, positional=True, required=True):
    """Adds a resource argument for a VMware admin cluster membership.

  Args:
    parser: The argparse parser to add the resource arg to.
    positional: bool, whether the argument is positional or not.
    required: bool, whether the argument is required or not.
  """
    name = 'admin_cluster_membership' if positional else '--admin-cluster-membership'
    admin_cluster_membership_help_text = 'membership of the admin cluster. Membership name is the same as the admin cluster name.\n\nExamples:\n\n  $ {command}\n        --admin-cluster-membership=projects/example-project-12345/locations/us-west1/memberships/example-admin-cluster-name\n\nor\n\n  $ {command}\n        --admin-cluster-membership-project=example-project-12345\n        --admin-cluster-membership-location=us-west1\n        --admin-cluster-membership=example-admin-cluster-name\n\n  '
    concept_parsers.ConceptParser.ForResource(name, GetAdminClusterMembershipResourceSpec(), admin_cluster_membership_help_text, required=required, flag_name_overrides={'project': '--admin-cluster-membership-project', 'location': '--admin-cluster-membership-location'}).AddToParser(parser)