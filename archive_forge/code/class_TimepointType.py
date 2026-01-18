from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class TimepointType(proto.Enum):
    """The type of timepoint information that is returned in the
        response.

        Values:
            TIMEPOINT_TYPE_UNSPECIFIED (0):
                Not specified. No timepoint information will
                be returned.
            SSML_MARK (1):
                Timepoint information of ``<mark>`` tags in SSML input will
                be returned.
        """
    TIMEPOINT_TYPE_UNSPECIFIED = 0
    SSML_MARK = 1