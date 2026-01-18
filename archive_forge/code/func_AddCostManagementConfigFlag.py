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
def AddCostManagementConfigFlag(parser, is_update=False):
    """Adds flags related to GKE cost management to the given parser."""
    help_text = '\nEnable the cost management feature.\n\nWhen enabled, you can get informational GKE cost breakdowns by cluster,\nnamespace and label in your billing data exported to BigQuery\n(https://cloud.google.com/billing/docs/how-to/export-data-bigquery).\n'
    if is_update:
        help_text += '\nUse --no-enable-cost-allocation to disable this feature.\n'
    parser.add_argument('--enable-cost-allocation', action='store_true', default=None, help=help_text)