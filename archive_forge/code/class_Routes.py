from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
@base.Hidden
class Routes(base.Group):
    """View your Cloud Run routes."""

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        return context