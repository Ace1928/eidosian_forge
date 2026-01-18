from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMacsecPreSharedKeyNameForRomoveKey(parser):
    """Adds keyName flag to the argparse.ArgumentParser."""
    parser.add_argument('--key-name', required=True, help='      The name of pre-shared key being removed from MACsec configuration of the\n      interconnect.\n      ')