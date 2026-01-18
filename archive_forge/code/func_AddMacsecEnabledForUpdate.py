from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMacsecEnabledForUpdate(parser):
    """Adds macsecEnabled flag to the argparse.ArgumentParser."""
    parser.add_argument('--enabled', action='store_true', default=None, help='      Enable or disable MACsec on this Interconnect. MACsec enablement will fail\n      if the MACsec configuration is not specified. Use --no-enabled to disable\n      it.\n      ')