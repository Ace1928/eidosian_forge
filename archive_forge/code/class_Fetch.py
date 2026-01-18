from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.config_management import utils
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import semver
@calliope_base.ReleaseTracks(calliope_base.ReleaseTrack.ALPHA)
class Fetch(base.DescribeCommand):
    """Prints the Config Management configuration applied to the given membership.

  The output is in the format that is used by the apply subcommand. The fields
  that have not been configured will be shown with default values.

  ## EXAMPLES

  To fetch the applied Config Management configuration, run:

    $ {command}
  """
    feature_name = 'configmanagement'

    @classmethod
    def Args(cls, parser):
        resources.AddMembershipResourceArg(parser)

    def Run(self, args):
        membership = base.ParseMembership(args, prompt=True, autoselect=True, search=True)
        f = self.GetFeature()
        membership_spec = None
        version = utils.get_backfill_version_from_feature(f, membership)
        for full_name, spec in self.hubclient.ToPyDict(f.membershipSpecs).items():
            if util.MembershipPartialName(full_name) == util.MembershipPartialName(membership) and spec is not None:
                membership_spec = spec.configmanagement
        if membership_spec is None:
            log.status.Print('Membership {} not initialized'.format(membership))
        template = yaml.load(utils.APPLY_SPEC_VERSION_1)
        full_config = template['spec']
        merge_config_sync(membership_spec, full_config, version)
        merge_policy_controller(membership_spec, full_config, version)
        merge_hierarchy_controller(membership_spec, full_config)
        return template