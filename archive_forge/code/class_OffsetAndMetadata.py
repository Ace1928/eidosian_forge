from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, Sequence, Tuple, TypeVar
from aiokafka.errors import KafkaError
class OffsetAndMetadata(NamedTuple):
    """The Kafka offset commit API

    The Kafka offset commit API allows users to provide additional metadata
    (in the form of a string) when an offset is committed. This can be useful
    (for example) to store information about which node made the commit,
    what time the commit was made, etc.
    """
    offset: int
    'The offset to be committed'
    metadata: str
    'Non-null metadata'