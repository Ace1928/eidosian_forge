from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddLoggingFlag(parser, autopilot=False):
    """Adds a --logging flag to parser."""
    help_text = 'Set the components that have logging enabled. Valid component values are:\n`SYSTEM`, `WORKLOAD`, `API_SERVER`, `CONTROLLER_MANAGER`, `SCHEDULER`, `NONE`\n\nFor more information, see\nhttps://cloud.google.com/stackdriver/docs/solutions/gke/installing#available-logs\n\nExamples:\n\n  $ {command} --logging=SYSTEM\n  $ {command} --logging=SYSTEM,API_SERVER,WORKLOAD\n  $ {command} --logging=NONE\n'
    if autopilot:
        help_text = 'Set the components that have logging enabled. Valid component values are:\n`SYSTEM`, `WORKLOAD`, `API_SERVER`, `CONTROLLER_MANAGER`, `SCHEDULER`\n\nThe default is `SYSTEM,WORKLOAD`. If this flag is set, then `SYSTEM` must be\nincluded.\n\nFor more information, see\nhttps://cloud.google.com/stackdriver/docs/solutions/gke/installing#available-logs\n\nExamples:\n\n  $ {command} --logging=SYSTEM\n  $ {command} --logging=SYSTEM,WORKLOAD\n  $ {command} --logging=SYSTEM,WORKLOAD,API_SERVER,CONTROLLER_MANAGER,SCHEDULER\n'
    parser.add_argument('--logging', type=arg_parsers.ArgList(), default=None, help=help_text, metavar='COMPONENT')