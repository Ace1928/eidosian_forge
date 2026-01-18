from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
def _CreateTimeSeriesListRequest(self, metrics_type):
    """Create a MonitoringProjectsTimeSeriesListRequest.

    Args:
      metrics_type: str, https://cloud.google.com/monitoring/api/metrics

    Returns:
      MonitoringProjectsTimeSeriesListRequest, input message for
      ProjectsTimeSeriesService List method.
    """
    current_time = datetime.datetime.utcnow()
    tp_proto_end_time = timestamp_pb2.Timestamp()
    tp_proto_end_time.FromDatetime(current_time)
    tp_proto_start_time = timestamp_pb2.Timestamp()
    tp_proto_start_time.FromDatetime(current_time - datetime.timedelta(seconds=600))
    return self.monitoring_message.MonitoringProjectsTimeSeriesListRequest(name='projects/{project}'.format(project=self.project.name), filter=FILTER_TEMPLATE.format(metrics_type=metrics_type, instance_name=self.instance.name), interval_endTime=tp_proto_end_time.ToJsonString(), interval_startTime=tp_proto_start_time.ToJsonString())