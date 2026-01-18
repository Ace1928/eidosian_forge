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
class CreateAppProfileBeta(CreateAppProfile):
    """Create a new Bigtable app profile."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create an app profile with a multi-cluster routing policy, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any\n\n          To create an app profile with a single-cluster routing policy which\n          routes all requests to `my-cluster-id`, run:\n\n            $ {command} my-single-cluster-app-profile --instance=my-instance-id --route-to=my-cluster-id\n\n          To create an app profile with a friendly description, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any --description="Routes requests for my use case"\n\n          To create an app profile with a request priority of PRIORITY_MEDIUM,\n          run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any --priority=PRIORITY_MEDIUM\n\n          To create an app profile with Data Boost enabled which bills usage to the host project, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --data-boost --data-boost-compute-billing-owner=HOST_PAYS\n\n          ')}

    @staticmethod
    def Args(parser):
        arguments.AddAppProfileResourceArg(parser, 'to create')
        arguments.ArgAdder(parser).AddDescription('app profile', required=False).AddAppProfileRouting().AddIsolation(allow_data_boost=True).AddForce('create')

    def _CreateAppProfile(self, app_profile_ref, args):
        """Creates an AppProfile with the given arguments.

    Args:
      app_profile_ref: A resource reference of the new app profile.
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ConflictingArgumentsException,
      OneOfArgumentsRequiredException:
        See app_profiles.Create(...)

    Returns:
      Created app profile resource object.
    """
        return app_profiles.Create(app_profile_ref, cluster=args.route_to, description=args.description, multi_cluster=args.route_any, restrict_to=args.restrict_to, transactional_writes=args.transactional_writes, priority=args.priority, data_boost=args.data_boost, data_boost_compute_billing_owner=args.data_boost_compute_billing_owner, force=args.force)