from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteSchemaRequest(proto.Message):
    """Request for the ``DeleteSchema`` method.

    Attributes:
        name (str):
            Required. Name of the schema to delete. Format is
            ``projects/{project}/schemas/{schema}``.
    """
    name: str = proto.Field(proto.STRING, number=1)