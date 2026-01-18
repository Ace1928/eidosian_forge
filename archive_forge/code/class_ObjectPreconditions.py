from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ObjectPreconditions(proto.Message):
    """Preconditions for a source object of a composition request.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                if_generation_match (int):
                    Only perform the composition if the
                    generation of the source object that would be
                    used matches this value.  If this value and a
                    generation are both specified, they must be the
                    same value or the call will fail.

                    This field is a member of `oneof`_ ``_if_generation_match``.
            """
    if_generation_match: int = proto.Field(proto.INT64, number=1, optional=True)