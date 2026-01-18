from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from surface.run.services import replace
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class MultiRegionReplace(replace.Replace):
    """Create or Update multi-region service from YAML."""

    def _ConnectionContext(self, args, region_label):
        return connection_context.GetConnectionContext(args, flags.Product.RUN, self.ReleaseTrack(), region_label=region_label, is_multiregion=True)

    def Run(self, args):
        if platforms.GetPlatform() != platforms.PLATFORM_MANAGED:
            raise c_exceptions.InvalidArgumentException('--platform', 'Multi-region Services are only supported on managed platform.')
        return super().Run(args)