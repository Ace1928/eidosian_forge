from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListHmacKeysResponse(proto.Message):
    """Hmac key list response with next page information.

    Attributes:
        hmac_keys (MutableSequence[googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.HmacKeyMetadata]):
            The list of items.
        next_page_token (str):
            The continuation token, used to page through
            large result sets. Provide this value in a
            subsequent request to return the next page of
            results.
    """

    @property
    def raw_page(self):
        return self
    hmac_keys: MutableSequence['HmacKeyMetadata'] = proto.RepeatedField(proto.MESSAGE, number=1, message='HmacKeyMetadata')
    next_page_token: str = proto.Field(proto.STRING, number=2)