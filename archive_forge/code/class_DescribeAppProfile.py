from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
class DescribeAppProfile(base.DescribeCommand):
    """Describe an existing Bigtable app profile."""
    detailed_help = {'EXAMPLES': textwrap.dedent("          To view an app profile's description, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id\n\n          ")}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.AddAppProfileResourceArg(parser, 'to describe')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        app_profile_ref = args.CONCEPTS.app_profile.Parse()
        return app_profiles.Describe(app_profile_ref)