from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import spanner_util
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Spanner(base.Group):
    """Manage your local Spanner emulator.

  This set of commands allows you to start and use a local Spanner emulator.
  """
    detailed_help = {'EXAMPLES': '          To start a local Cloud Spanner emulator, run:\n\n            $ {command} start\n          '}

    def Filter(self, context, args):
        current_os = platforms.OperatingSystem.Current()
        if current_os is platforms.OperatingSystem.LINUX:
            util.EnsureComponentIsInstalled(spanner_util.SPANNER_EMULATOR_COMPONENT_ID, spanner_util.SPANNER_EMULATOR_TITLE)
        else:
            _RequireDockerInstalled()