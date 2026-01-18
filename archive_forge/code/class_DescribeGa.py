from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import apikeys
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.services import common_flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DescribeGa(base.DescribeCommand):
    """Describe an API key's metadata."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        common_flags.key_flag(parser=parser, suffix='to describe', api_version='v2')

    def Run(self, args):
        """Run command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      The metadata of API key.
    """
        client = apikeys.GetClientInstance(self.ReleaseTrack())
        messages = client.MESSAGES_MODULE
        key_ref = args.CONCEPTS.key.Parse()
        request = messages.ApikeysProjectsLocationsKeysGetRequest(name=key_ref.RelativeName())
        return client.projects_locations_keys.Get(request)