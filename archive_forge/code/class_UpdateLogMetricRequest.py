from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import distribution_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class UpdateLogMetricRequest(proto.Message):
    """The parameters to UpdateLogMetric.

    Attributes:
        metric_name (str):
            Required. The resource name of the metric to update:

            ::

                "projects/[PROJECT_ID]/metrics/[METRIC_ID]"

            The updated metric must be provided in the request and it's
            ``name`` field must be the same as ``[METRIC_ID]`` If the
            metric does not exist in ``[PROJECT_ID]``, then a new metric
            is created.
        metric (googlecloudsdk.generated_clients.gapic_clients.logging_v2.types.LogMetric):
            Required. The updated metric.
    """
    metric_name: str = proto.Field(proto.STRING, number=1)
    metric: 'LogMetric' = proto.Field(proto.MESSAGE, number=2, message='LogMetric')