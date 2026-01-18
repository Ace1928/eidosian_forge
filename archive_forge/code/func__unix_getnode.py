import os
import sys
from enum import Enum, _simple_enum
def _unix_getnode():
    """Get the hardware address on Unix using the _uuid extension module."""
    if _generate_time_safe:
        uuid_time, _ = _generate_time_safe()
        return UUID(bytes=uuid_time).node