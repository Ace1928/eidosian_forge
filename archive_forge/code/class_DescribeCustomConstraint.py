from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DescribeCustomConstraint(base.DescribeCommand):
    """Describe a custom constraint."""

    @staticmethod
    def Args(parser):
        arguments.AddCustomConstraintArgToParser(parser)
        arguments.AddOrganizationResourceFlagsToParser(parser)

    def Run(self, args):
        """Gets the custom constraint.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
       The retrieved custom constraint.
    """
        org_policy_api = org_policy_service.OrgPolicyApi(self.ReleaseTrack())
        custom_constraint_name = utils.GetCustomConstraintFromArgs(args)
        return org_policy_api.GetCustomConstraint(custom_constraint_name)