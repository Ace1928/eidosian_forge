from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.command_lib.container import container_command_util as cmd_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _AddAdditionalZonesFlag(parser, deprecated=True):
    action = None
    if deprecated:
        action = actions.DeprecationAction('additional-zones', warn='This flag is deprecated. Use --node-locations=PRIMARY_ZONE,[ZONE,...] instead.')
    parser.add_argument('--additional-zones', type=arg_parsers.ArgList(min_length=1), action=action, metavar='ZONE', help="The set of additional zones in which the specified node footprint should be\nreplicated. All zones must be in the same region as the cluster's primary zone.\nIf additional-zones is not specified, all nodes will be in the cluster's primary\nzone.\n\nNote that `NUM_NODES` nodes will be created in each zone, such that if you\nspecify `--num-nodes=4` and choose one additional zone, 8 nodes will be created.\n\nMultiple locations can be specified, separated by commas. For example:\n\n  $ {command} example-cluster --zone us-central1-a --additional-zones us-central1-b,us-central1-c\n")