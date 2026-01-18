from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
def AddConcurrentControlGroupArgs(parent_group):
    concurrent_group = parent_group.add_argument_group('  Concurrent Maintenance Controls Group. Defines a group config that, when\n  attached to an instance, recognizes that instance as a part of a group of\n  instances where only up the configured amount of instances in that group can\n  undergo simultaneous maintenance.\n  ')
    concurrent_group.add_argument('--concurrency-limit-percent', type=int, help='  Defines the max percentage of instances in a concurrency group that go to\n  maintenance simultaneously. Value must be greater or equal to 1 and less or\n  equal to 100.\n  Usage examples:\n  `--concurrency-limit=1` sets to 1%.\n  `--concurrency-limit=55` sets to 55%.')