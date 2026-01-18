from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import connection_profiles
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datastream import resource_args
from googlecloudsdk.command_lib.datastream.connection_profiles import flags as cp_flags
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
class _Discover:
    """Base class for discovering Datastream connection profiles."""
    detailed_help = {'DESCRIPTION': DESCRIPTION, 'EXAMPLES': EXAMPLES}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        concept_parsers.ConceptParser.ForResource('--location', resource_args.GetLocationResourceSpec(), group_help='The location you want to list the connection profiles for.', required=True).AddToParser(parser)
        resource_args.AddConnectionProfileDiscoverResourceArg(parser)
        cp_flags.AddDepthGroup(parser)
        cp_flags.AddRdbmsGroup(parser)
        cp_flags.AddHierarchyGroup(parser)

    def Run(self, args):
        """Discover a Datastream connection profile.

    Args:
      args: argparse.Namespace, The arguments that this command was invoked
        with.

    Returns:
      A dict object representing the operations resource describing the discover
      operation if the discover was successful.
    """
        project = properties.VALUES.core.project.Get(required=True)
        location = args.location
        parent_ref = util.ParentRef(project, location)
        cp_client = connection_profiles.ConnectionProfilesClient()
        return cp_client.Discover(parent_ref, self.ReleaseTrack(), args)