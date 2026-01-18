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
def AddFleetProjectFlag(parser, is_update=False):
    """Adds fleet related flags to the parser."""
    enable_text = '\nSets fleet host project for the cluster. If specified, the current cluster will be registered as a fleet membership under the fleet host project.\n\nExample:\n$ {command} --fleet-project=my-project\n'
    auto_enable_text = '\nSet cluster project as the fleet host project. This will register the cluster to the same project.\nTo register the cluster to a fleet in a different project, please use `--fleet-project=FLEET_HOST_PROJECT`.\nExample:\n$ {command} --enable-fleet\n'
    unset_text = '\nRemove the cluster from current fleet host project.\nExample:\n$ {command} --clear-fleet-project\n'
    parser.add_argument('--fleet-project', help=enable_text, metavar='PROJECT_ID_OR_NUMBER', type=str)
    parser.add_argument('--enable-fleet', default=None, help=auto_enable_text, action='store_true')
    if is_update:
        parser.add_argument('--clear-fleet-project', help=unset_text, default=None, action='store_true')