from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class Autoclass(proto.Message):
    """Configuration for a bucket's Autoclass feature.

        .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

        Attributes:
            enabled (bool):
                Enables Autoclass.
            toggle_time (google.protobuf.timestamp_pb2.Timestamp):
                Output only. Latest instant at which the ``enabled`` field
                was set to true after being disabled/unconfigured or set to
                false after being enabled. If Autoclass is enabled when the
                bucket is created, the toggle_time is set to the bucket
                creation time.
            terminal_storage_class (str):
                An object in an Autoclass bucket will
                eventually cool down to the terminal storage
                class if there is no access to the object. The
                only valid values are NEARLINE and ARCHIVE.

                This field is a member of `oneof`_ ``_terminal_storage_class``.
            terminal_storage_class_update_time (google.protobuf.timestamp_pb2.Timestamp):
                Output only. Latest instant at which the
                autoclass terminal storage class was updated.

                This field is a member of `oneof`_ ``_terminal_storage_class_update_time``.
        """
    enabled: bool = proto.Field(proto.BOOL, number=1)
    toggle_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    terminal_storage_class: str = proto.Field(proto.STRING, number=3, optional=True)
    terminal_storage_class_update_time: timestamp_pb2.Timestamp = proto.Field(proto.MESSAGE, number=4, optional=True, message=timestamp_pb2.Timestamp)