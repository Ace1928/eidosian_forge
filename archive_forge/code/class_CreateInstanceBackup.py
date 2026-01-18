from googlecloudsdk.api_lib.looker import backups
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.looker import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateInstanceBackup(base.CreateCommand):
    """Create a backup of a Looker instance."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddInstance(parser)
        parser.add_argument('--region', required=True, help='             The name of the Looker region of the instance. Overrides the\n            default looker/region property value for this command invocation.\n            ')

    def Run(self, args):
        parent = resources.REGISTRY.Parse(args.instance, params={'projectsId': properties.VALUES.core.project.GetOrFail, 'locationsId': args.region}, api_version=backups.API_VERSION_DEFAULT, collection='looker.projects.locations.instances').RelativeName()
        return backups.CreateBackup(parent)