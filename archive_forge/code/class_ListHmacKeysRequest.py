from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListHmacKeysRequest(proto.Message):
    """Request to fetch a list of HMAC keys under a given project.

    Attributes:
        project (str):
            Required. The project to list HMAC keys for,
            in the format of "projects/{projectIdentifier}".
            {projectIdentifier} can be the project ID or
            project number.
        page_size (int):
            The maximum number of keys to return.
        page_token (str):
            A previously returned token from
            ListHmacKeysResponse to get the next page.
        service_account_email (str):
            If set, filters to only return HMAC keys for
            specified service account.
        show_deleted_keys (bool):
            If set, return deleted keys that have not yet
            been wiped out.
    """
    project: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    service_account_email: str = proto.Field(proto.STRING, number=4)
    show_deleted_keys: bool = proto.Field(proto.BOOL, number=5)