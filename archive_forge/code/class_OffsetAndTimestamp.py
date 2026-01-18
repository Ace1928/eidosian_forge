from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class OffsetAndTimestamp(NamedTuple):
    offset: int
    timestamp: Optional[int]