from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ReadObjectResponse(proto.Message):
    """Response message for ReadObject.

    Attributes:
        checksummed_data (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ChecksummedData):
            A portion of the data for the object. The service **may**
            leave ``data`` empty for any given ``ReadResponse``. This
            enables the service to inform the client that the request is
            still live while it is running an operation to generate more
            data.
        object_checksums (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ObjectChecksums):
            The checksums of the complete object. If the
            object is downloaded in full, the client should
            compute one of these checksums over the
            downloaded object and compare it against the
            value provided here.
        content_range (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ContentRange):
            If read_offset and or read_limit was specified on the
            ReadObjectRequest, ContentRange will be populated on the
            first ReadObjectResponse message of the read stream.
        metadata (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            Metadata of the object whose media is being
            returned. Only populated in the first response
            in the stream.
    """
    checksummed_data: 'ChecksummedData' = proto.Field(proto.MESSAGE, number=1, message='ChecksummedData')
    object_checksums: 'ObjectChecksums' = proto.Field(proto.MESSAGE, number=2, message='ObjectChecksums')
    content_range: 'ContentRange' = proto.Field(proto.MESSAGE, number=3, message='ContentRange')
    metadata: 'Object' = proto.Field(proto.MESSAGE, number=4, message='Object')