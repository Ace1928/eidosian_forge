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
def AddMonitoringFlag(parser, autopilot=False):
    """Adds a --monitoring flag to parser."""
    help_text = 'Set the components that have monitoring enabled. Valid component values are:\n`SYSTEM`, `WORKLOAD` (Deprecated), `NONE`, `API_SERVER`, `CONTROLLER_MANAGER`,\n`SCHEDULER`, `DAEMONSET`, `DEPLOYMENT`, `HPA`, `POD`, `STATEFULSET`, `STORAGE`\n\nFor more information, see\nhttps://cloud.google.com/stackdriver/docs/solutions/gke/installing#available-metrics\n\nExamples:\n\n  $ {command} --monitoring=SYSTEM,API_SERVER,POD\n  $ {command} --monitoring=NONE\n'
    if autopilot:
        help_text = 'Set the components that have monitoring enabled. Valid component values are:\n`SYSTEM`, `WORKLOAD` (Deprecated), `NONE`, `API_SERVER`, `CONTROLLER_MANAGER`,\n`SCHEDULER`, `DAEMONSET`, `DEPLOYMENT`, `HPA`, `POD`, `STATEFULSET`, `STORAGE`\n\nFor more information, see\nhttps://cloud.google.com/stackdriver/docs/solutions/gke/installing#available-metrics\n\nExamples:\n\n  $ {command} --monitoring=SYSTEM,API_SERVER,POD\n  $ {command} --monitoring=SYSTEM\n'
    parser.add_argument('--monitoring', type=arg_parsers.ArgList(), default=None, help=help_text, metavar='COMPONENT')