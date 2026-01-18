import enum
from typing import Optional, List, Union, Iterable, Tuple
class DurationUnit(enum.Enum):
    """Valid values for the unit of durations."""
    NANOSECOND = 'ns'
    MICROSECOND = 'us'
    MILLISECOND = 'ms'
    SECOND = 's'
    SAMPLE = 'dt'