from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class FilterWriteMode(proto.Enum):
    """Behavior to apply to the built-in ``_Default`` sink inclusion
            filter.

            Values:
                FILTER_WRITE_MODE_UNSPECIFIED (0):
                    The filter's write mode is unspecified. This
                    mode must not be used.
                APPEND (1):
                    The contents of ``filter`` will be appended to the built-in
                    ``_Default`` sink filter. Using the append mode with an
                    empty filter will keep the sink inclusion filter unchanged.
                OVERWRITE (2):
                    The contents of ``filter`` will overwrite the built-in
                    ``_Default`` sink filter.
            """
    FILTER_WRITE_MODE_UNSPECIFIED = 0
    APPEND = 1
    OVERWRITE = 2