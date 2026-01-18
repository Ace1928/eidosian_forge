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
def AddResourceUsageExportFlags(parser, is_update=False, hidden=False):
    """Adds flags about exporting cluster resource usage to BigQuery."""
    group = parser.add_group("Exports cluster's usage of cloud resources", hidden=hidden)
    if is_update:
        group.is_mutex = True
        group.add_argument('--clear-resource-usage-bigquery-dataset', action='store_true', hidden=hidden, default=None, help='Disables exporting cluster resource usage to BigQuery.')
        group = group.add_group()
    dataset_help_text = "The name of the BigQuery dataset to which the cluster's usage of cloud\nresources is exported. A table will be created in the specified dataset to\nstore cluster resource usage. The resulting table can be joined with BigQuery\nBilling Export to produce a fine-grained cost breakdown.\n\nExamples:\n\n  $ {command} example-cluster --resource-usage-bigquery-dataset=example_bigquery_dataset_name\n"
    group.add_argument('--resource-usage-bigquery-dataset', default=None, hidden=hidden, help=dataset_help_text)
    network_egress_help_text = 'Enable network egress metering on this cluster.\n\nWhen enabled, a DaemonSet is deployed into the cluster. Each DaemonSet pod\nmeters network egress traffic by collecting data from the conntrack table, and\nexports the metered metrics to the specified destination.\n\nNetwork egress metering is disabled if this flag is omitted, or when\n`--no-enable-network-egress-metering` is set.\n'
    group.add_argument('--enable-network-egress-metering', action='store_true', default=None, hidden=hidden, help=network_egress_help_text)
    resource_consumption_help_text = 'Enable resource consumption metering on this cluster.\n\nWhen enabled, a table will be created in the specified BigQuery dataset to store\nresource consumption data. The resulting table can be joined with the resource\nusage table or with BigQuery billing export.\n\nResource consumption metering is enabled unless `--no-enable-resource-\nconsumption-metering` is set.\n'
    if is_update:
        resource_consumption_help_text = 'Enable resource consumption metering on this cluster.\n\nWhen enabled, a table will be created in the specified BigQuery dataset to store\nresource consumption data. The resulting table can be joined with the resource\nusage table or with BigQuery billing export.\n\nTo disable resource consumption metering, set `--no-enable-resource-consumption-\nmetering`. If this flag is omitted, then resource consumption metering will\nremain enabled or disabled depending on what is already configured for this\ncluster.\n'
    group.add_argument('--enable-resource-consumption-metering', action='store_true', default=None, hidden=hidden, help=resource_consumption_help_text)