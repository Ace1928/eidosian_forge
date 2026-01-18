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
def _SetDataprocMetricConfig(args, cluster_config, dataproc):
    """Method to set Metric source and the corresponding optional overrides to DataprocMetricConfig.

  Metric overrides can be read from either metric-overrides or
  metric-overrides-file argument.
  We do basic validation on metric-overrides :
  * Ensure that all entries of metric-overrides are prefixed with camel case of
  the metric source.
    Example :
    "sparkHistoryServer:JVM:Memory:NonHeapMemoryUsage.used" is valid metric
    override for the metric-source spark-history-server
    but "spark-history-server:JVM:Memory:NonHeapMemoryUsage.used" is not.
  * Metric overrides are passed only for the metric sources enabled via
  args.metric_sources.

  Args:
    args: arguments passed to create cluster command.
    cluster_config: cluster configuration to be updated with
      DataprocMetricConfig.
    dataproc: Dataproc API definition.
  """

    def _GetCamelCaseMetricSource(ms):
        title_case = ms.lower().title().replace('_', '').replace('-', '')
        return title_case[0].lower() + title_case[1:]
    metric_source_to_overrides_dict = dict()
    metric_overrides = [m.strip() for m in _GetMetricOverrides(args) if m.strip()]
    if metric_overrides:
        invalid_metric_overrides = []
        valid_metric_prefixes = [_GetCamelCaseMetricSource(ms) for ms in args.metric_sources]
        for metric in metric_overrides:
            prefix = metric.split(':')[0]
            if prefix not in valid_metric_prefixes:
                invalid_metric_overrides.append(metric)
            metric_source_to_overrides_dict.setdefault(prefix, []).append(metric)
        if invalid_metric_overrides:
            raise exceptions.ArgumentError('Found invalid metric overrides: ' + ','.join(invalid_metric_overrides) + '. Please ensure the metric overrides only have the following prefixes that correspond to the metric-sources that are enabled: ' + ','.join(valid_metric_prefixes))
    cluster_config.dataprocMetricConfig = dataproc.messages.DataprocMetricConfig(metrics=[])
    for metric_source in args.metric_sources:
        metric_source_in_camel_case = _GetCamelCaseMetricSource(metric_source)
        metric_overrides = metric_source_to_overrides_dict.get(metric_source_in_camel_case, [])
        cluster_config.dataprocMetricConfig.metrics.append(dataproc.messages.Metric(metricSource=arg_utils.ChoiceToEnum(metric_source, dataproc.messages.Metric.MetricSourceValueValuesEnum), metricOverrides=metric_overrides))