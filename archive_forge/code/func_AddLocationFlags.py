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
def AddLocationFlags(parser):
    """Adds the --location, --zone, and --region flags to the parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--location', help='Compute zone or region (e.g. us-central1-a or us-central1) for the cluster.')
    group.add_argument('--zone', '-z', help='Compute zone (e.g. us-central1-a) for the cluster.', action=actions.StoreProperty(properties.VALUES.compute.zone))
    group.add_argument('--region', help='Compute region (e.g. us-central1) for the cluster.')