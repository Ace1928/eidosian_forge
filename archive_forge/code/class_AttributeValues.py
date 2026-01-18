from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class AttributeValues(proto.Message):
    """The values associated with a key of an attribute.

    Attributes:
        values (MutableSequence[bytes]):
            The list of values associated with a key.
    """
    values: MutableSequence[bytes] = proto.RepeatedField(proto.BYTES, number=1)