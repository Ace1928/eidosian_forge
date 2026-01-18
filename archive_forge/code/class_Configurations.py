from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import flags
@base.Hidden
class Configurations(base.Group):
    """View and manage your Cloud Run configurations."""
    detailed_help = {'EXAMPLES': '\n          To describe the configuration managed by the service foo:\n\n            $ {command} describe foo\n\n      '}

    @staticmethod
    def Args(parser):
        """Adds --platform and the various related args."""
        flags.AddPlatformAndLocationFlags(parser)

    def Filter(self, context, args):
        """Runs before command.Run and validates platform with passed args."""
        flags.GetAndValidatePlatform(args, self.ReleaseTrack(), flags.Product.RUN)
        return context