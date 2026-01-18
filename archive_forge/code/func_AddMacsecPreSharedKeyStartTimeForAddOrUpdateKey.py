from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def AddMacsecPreSharedKeyStartTimeForAddOrUpdateKey(parser):
    """Adds keyName flag to the argparse.ArgumentParser."""
    parser.add_argument('--start-time', required=False, default=None, help='      A RFC3339 timestamp on or after which the key is valid. startTime can be\n      in the future. If the keychain has a single key, --start-time can be\n      omitted. If the keychain has multiple keys, --start-time is mandatory for\n      each key. The start times of two consecutive keys must be at least 6 hours\n      apart.\n      ')