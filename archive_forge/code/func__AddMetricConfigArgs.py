from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def _AddMetricConfigArgs(parser, dataproc):
    """Adds DataprocMetricConfig related args to the parser."""
    metric_overrides_detailed_help = '\n  List of metrics that override the default metrics enabled for the metric\n  sources. Any of the\n  [available OSS metrics](https://cloud.google.com/dataproc/docs/guides/monitoring#available_oss_metrics)\n  and all Spark metrics, can be listed for collection as a metric override.\n  Override metric values are case sensitive, and must be provided, if\n  appropriate, in CamelCase format, for example:\n\n  *sparkHistoryServer:JVM:Memory:NonHeapMemoryUsage.committed*\n  *hiveserver2:JVM:Memory:NonHeapMemoryUsage.used*\n\n  Only the specified overridden metrics will be collected from a given metric\n  source. For example, if one or more *spark:executive* metrics are listed as\n  metric overrides, other *SPARK* metrics will not be collected. The collection\n  of default OSS metrics from other metric sources is unaffected. For example,\n  if both *SPARK* and *YARN* metric sources are enabled, and overrides are\n  provided for Spark metrics only, all default YARN metrics will be collected.\n\n  The source of the specified metric override must be enabled. For example,\n  if one or more *spark:driver* metrics are provided as metric overrides,\n  the spark metric source must be enabled (*--metric-sources=spark*).\n  '
    metric_config_group = parser.add_group()
    metric_config_group.add_argument('--metric-sources', metavar='METRIC_SOURCE', type=arg_parsers.ArgList(arg_utils.ChoiceToEnumName, choices=_GetValidMetricSourceChoices(dataproc)), required=True, help='Specifies a list of cluster [Metric Sources](https://cloud.google.com/dataproc/docs/guides/monitoring#available_oss_metrics) to collect custom metrics.')
    metric_overrides_group = metric_config_group.add_mutually_exclusive_group()
    metric_overrides_group.add_argument('--metric-overrides', type=arg_parsers.ArgList(), action=arg_parsers.UpdateAction, metavar='METRIC_SOURCE:INSTANCE:GROUP:METRIC', help=metric_overrides_detailed_help)
    metric_overrides_group.add_argument('--metric-overrides-file', help='      Path to a file containing list of Metrics that override the default metrics enabled for the metric sources.\n      The path can be a Cloud Storage URL (example: `gs://path/to/file`) or a local file system path.\n      ')