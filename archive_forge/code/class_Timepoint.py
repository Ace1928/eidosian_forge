from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class Timepoint(proto.Message):
    """This contains a mapping between a certain point in the input
    text and a corresponding time in the output audio.

    Attributes:
        mark_name (str):
            Timepoint name as received from the client within ``<mark>``
            tag.
        time_seconds (float):
            Time offset in seconds from the start of the
            synthesized audio.
    """
    mark_name: str = proto.Field(proto.STRING, number=4)
    time_seconds: float = proto.Field(proto.DOUBLE, number=3)