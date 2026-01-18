from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class UniformBucketLevelAccess(proto.Message):
    """Settings for Uniform Bucket level access.
            See
            https://cloud.google.com/storage/docs/uniform-bucket-level-access.

            Attributes:
                enabled (bool):
                    If set, access checks only use bucket-level
                    IAM policies or above.
                lock_time (google.protobuf.timestamp_pb2.Timestamp):
                    The deadline time for changing
                    ``iam_config.uniform_bucket_level_access.enabled`` from
                    ``true`` to ``false``. Mutable until the specified deadline
                    is reached, but not afterward.
            """
    enabled: bool = proto.Field(proto.BOOL, number=1)
    lock_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)