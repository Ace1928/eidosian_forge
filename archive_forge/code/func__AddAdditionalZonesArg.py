from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from six.moves import input  # pylint: disable=redefined-builtin
def _AddAdditionalZonesArg(mutex_group, deprecated=True):
    action = None
    if deprecated:
        action = actions.DeprecationAction('additional-zones', warn='This flag is deprecated. Use --node-locations=PRIMARY_ZONE,[ZONE,...] instead.')
    mutex_group.add_argument('--additional-zones', type=arg_parsers.ArgList(), action=action, metavar='ZONE', help='The set of additional zones in which the cluster\'s node footprint should be\nreplicated. All zones must be in the same region as the cluster\'s primary zone.\n\nNote that the exact same footprint will be replicated in all zones, such that\nif you created a cluster with 4 nodes in a single zone and then use this option\nto spread across 2 more zones, 8 additional nodes will be created.\n\nMultiple locations can be specified, separated by commas. For example:\n\n  $ {command} example-cluster --zone us-central1-a --additional-zones us-central1-b,us-central1-c\n\nTo remove all zones other than the cluster\'s primary zone, pass the empty string\nto the flag. For example:\n\n  $ {command} example-cluster --zone us-central1-a --additional-zones ""\n')