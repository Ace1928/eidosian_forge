from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class UpdateAppProfileBeta(UpdateAppProfile):
    """Update a Bigtable app profile."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To update an app profile to use a multi-cluster routing policy, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any\n\n          To update an app profile to use a single-cluster routing policy that\n          routes all requests to `my-cluster-id` and allows transactional\n          writes, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-to=my-cluster-id --transactional-writes\n\n          To update the description for an app profile, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --description="New description"\n\n          To update the request priority for an app profile to PRIORITY_LOW, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --priority=PRIORITY_LOW\n\n          To update an app profile to enable Data Boost which bills usage to the host project, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --data-boost --data-boost-compute-billing-owner=HOST_PAYS\n\n          ')}

    @staticmethod
    def Args(parser):
        arguments.AddAppProfileResourceArg(parser, 'to update')
        arguments.ArgAdder(parser).AddDescription('app profile', required=False).AddAppProfileRouting(required=False).AddIsolation(allow_data_boost=True).AddForce('update').AddAsync()

    def _UpdateAppProfile(self, app_profile_ref, args):
        """Updates an AppProfile with the given arguments.

    Args:
      app_profile_ref: A resource reference of the new app profile.
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ConflictingArgumentsException,
      OneOfArgumentsRequiredException:
        See app_profiles.Update(...)

    Returns:
      Long running operation.
    """
        return app_profiles.Update(app_profile_ref, cluster=args.route_to, description=args.description, multi_cluster=args.route_any, restrict_to=args.restrict_to, transactional_writes=args.transactional_writes, priority=args.priority, data_boost=args.data_boost, data_boost_compute_billing_owner=args.data_boost_compute_billing_owner, force=args.force)