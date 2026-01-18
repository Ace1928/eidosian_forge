from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class GetNotificationConfigRequest(proto.Message):
    """Request message for GetNotificationConfig.

    Attributes:
        name (str):
            Required. The parent bucket of the NotificationConfig.
            Format:
            ``projects/{project}/buckets/{bucket}/notificationConfigs/{notificationConfig}``
    """
    name: str = proto.Field(proto.STRING, number=1)