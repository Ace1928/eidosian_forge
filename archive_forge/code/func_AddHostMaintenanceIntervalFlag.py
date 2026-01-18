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
def AddHostMaintenanceIntervalFlag(parser, for_node_pool=False, hidden=True):
    """Adds a --host-maintenance-interval flag to the given parser."""
    type_validator = arg_parsers.RegexpValidator('^(PERIODIC|AS_NEEDED)$', 'Type must be either"PERIODIC" or "AS_NEEDED"')
    if for_node_pool:
        help_text = "Specify the frequency of planned host maintenance events in the new nodepool\n\nExamples:\n\n  $ {command} node-pool-1 example-cluster --host-maintenance-interval=PERIODIC\n\nThe maintenance interval type must be either 'PERIODIC' or 'AS_NEEDED'\n"
    else:
        help_text = "Specify the frequency of planned maintenance events in the new cluster\n\nExamples:\n\n  $ {command} example-cluster --host-maintenance-interval=PERIODIC\n\nThe maintenance interval type must be either 'PERIODIC' or 'AS_NEEDED'\n"
    parser.add_argument('--host-maintenance-interval', type=type_validator, hidden=hidden, help=help_text)