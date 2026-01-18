from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CommonObjectRequestParams(proto.Message):
    """Parameters that can be passed to any object request.

    Attributes:
        encryption_algorithm (str):
            Encryption algorithm used with the
            Customer-Supplied Encryption Keys feature.
        encryption_key_bytes (bytes):
            Encryption key used with the
            Customer-Supplied Encryption Keys feature. In
            raw bytes format (not base64-encoded).
        encryption_key_sha256_bytes (bytes):
            SHA256 hash of encryption key used with the
            Customer-Supplied Encryption Keys feature.
    """
    encryption_algorithm: str = proto.Field(proto.STRING, number=1)
    encryption_key_bytes: bytes = proto.Field(proto.BYTES, number=4)
    encryption_key_sha256_bytes: bytes = proto.Field(proto.BYTES, number=5)