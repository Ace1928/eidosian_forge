from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
class Unmanage(base.UpdateCommand):
    """Remove the Config Management Feature Spec for the given membership.

  Remove the Config Management Feature Spec for the given membership. The
  existing ConfigManagement resources in the clusters will become unmanaged.

  ## EXAMPLES

  To remove the Config Management Feature spec for a membership, run:

    $ {command} --membership=MEMBERSHIP_NAME
  """
    feature_name = 'configmanagement'

    @classmethod
    def Args(cls, parser):
        resources.AddMembershipResourceArg(parser)

    def Run(self, args):
        membership = base.ParseMembership(args, prompt=True, autoselect=True, search=True)
        membership_key = membership
        specs = {membership_key: self.messages.MembershipFeatureSpec()}
        patch = self.messages.Feature(membershipSpecs=self.hubclient.ToMembershipSpecs(specs))
        self.Update(['membership_specs'], patch)