from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class SoftDeletePolicy(proto.Message):
    """Soft delete policy properties of a bucket.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            retention_duration (google.protobuf.duration_pb2.Duration):
                The period of time that soft-deleted objects
                in the bucket must be retained and cannot be
                permanently deleted. The duration must be
                greater than or equal to 7 days and less than 1
                year.

                This field is a member of `oneof`_ ``_retention_duration``.
            effective_time (google.protobuf.timestamp_pb2.Timestamp):
                Time from which the policy was effective.
                This is service-provided.

                This field is a member of `oneof`_ ``_effective_time``.
        """
    retention_duration: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=1, optional=True, message=duration_pb2.Duration)
    effective_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=2, optional=True, message=timestamp_pb2.Timestamp)