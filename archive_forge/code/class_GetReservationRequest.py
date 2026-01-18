from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class GetReservationRequest(proto.Message):
    """Request for GetReservation.

    Attributes:
        name (str):
            Required. The name of the reservation whose configuration to
            return. Structured like:
            projects/{project_number}/locations/{location}/reservations/{reservation_id}
    """
    name: str = proto.Field(proto.STRING, number=1)