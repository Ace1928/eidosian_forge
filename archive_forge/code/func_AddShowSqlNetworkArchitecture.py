from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util import completers
def AddShowSqlNetworkArchitecture(parser):
    """Adds the `--show-sql-network-architecture` flag to the parser."""
    kwargs = _GetKwargsForBoolFlag(False)
    parser.add_argument('--show-sql-network-architecture', required=False, help="Show the instance's current SqlNetworkArchitecture backend in addition\n        to the default output list. An instance could use either the old or new\n        network architecture. The new network architecture offers better\n        isolation, reliability, and faster new feature adoption.", **kwargs)