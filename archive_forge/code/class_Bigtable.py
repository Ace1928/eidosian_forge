from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.emulators import bigtable_util
from googlecloudsdk.command_lib.emulators import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import platforms
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Bigtable(base.Group):
    """Manage your local Bigtable emulator.

  This set of commands allows you to start and use a local Bigtable emulator.
  """
    detailed_help = {'EXAMPLES': '          To start a local Bigtable emulator, run:\n\n            $ {command} start\n          '}

    def Filter(self, context, args):
        util.EnsureComponentIsInstalled(bigtable_util.BIGTABLE, bigtable_util.BIGTABLE_TITLE)