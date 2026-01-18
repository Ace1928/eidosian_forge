from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.org_policies import arguments
from googlecloudsdk.command_lib.org_policies import utils
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DeleteCustomConstraint(base.DeleteCommand):
    """Deletes a custom constraint."""

    @staticmethod
    def Args(parser):
        arguments.AddCustomConstraintArgToParser(parser)
        arguments.AddOrganizationResourceFlagsToParser(parser)

    def Run(self, args):
        """Deletes the custom constraint.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
       If the custom constraint is deleted, then messages.GoogleProtobufEmpty.
    """
        org_policy_api = org_policy_service.OrgPolicyApi(self.ReleaseTrack())
        custom_constraint_name = utils.GetCustomConstraintFromArgs(args)
        delete_response = org_policy_api.DeleteCustomConstraint(custom_constraint_name)
        log.DeletedResource(custom_constraint_name, 'custom constraint')
        return delete_response