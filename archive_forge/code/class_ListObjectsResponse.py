from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListObjectsResponse(proto.Message):
    """The result of a call to Objects.ListObjects

    Attributes:
        objects (MutableSequence[googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object]):
            The list of items.
        prefixes (MutableSequence[str]):
            The list of prefixes of objects
            matching-but-not-listed up to and including the
            requested delimiter.
        next_page_token (str):
            The continuation token, used to page through
            large result sets. Provide this value in a
            subsequent request to return the next page of
            results.
    """

    @property
    def raw_page(self):
        return self
    objects: MutableSequence['Object'] = proto.RepeatedField(proto.MESSAGE, number=1, message='Object')
    prefixes: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=2)
    next_page_token: str = proto.Field(proto.STRING, number=3)