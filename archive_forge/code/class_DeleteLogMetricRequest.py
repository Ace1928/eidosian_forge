from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.api import distribution_pb2  # type: ignore
from google.api import metric_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteLogMetricRequest(proto.Message):
    """The parameters to DeleteLogMetric.

    Attributes:
        metric_name (str):
            Required. The resource name of the metric to delete:

            ::

                "projects/[PROJECT_ID]/metrics/[METRIC_ID]".
    """
    metric_name: str = proto.Field(proto.STRING, number=1)