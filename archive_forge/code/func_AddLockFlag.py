from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.functions.v1 import util as functions_api_util
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddLockFlag(parser, hidden=False):
    """Add --lock-id flag."""
    parser.add_argument('--lock-id', required=True, hidden=hidden, help='Lock ID of the lock file to verify person importing owns lock.')