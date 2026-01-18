from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ObjectChecksums(proto.Message):
    """Message used for storing full (not subrange) object
    checksums.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        crc32c (int):
            CRC32C digest of the object data. Computed by
            the Cloud Storage service for all written
            objects. If set in a WriteObjectRequest, service
            will validate that the stored object matches
            this checksum.

            This field is a member of `oneof`_ ``_crc32c``.
        md5_hash (bytes):
            128 bit MD5 hash of the object data. For more information
            about using the MD5 hash, see
            [https://cloud.google.com/storage/docs/hashes-etags#json-api][Hashes
            and ETags: Best Practices]. Not all objects will provide an
            MD5 hash. For example, composite objects provide only crc32c
            hashes. This value is equivalent to running
            ``cat object.txt | openssl md5 -binary``
    """
    crc32c: int = proto.Field(proto.FIXED32, number=1, optional=True)
    md5_hash: bytes = proto.Field(proto.BYTES, number=2)