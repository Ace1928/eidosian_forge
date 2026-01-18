from abc import abstractmethod, ABCMeta
from typing import Generic, List, NamedTuple
import asyncio
from google.cloud.pubsublite.internal.wire.connection import Request, Response
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
class BatchSize(NamedTuple):
    element_count: int
    byte_count: int

    def __add__(self, other: 'BatchSize') -> 'BatchSize':
        return BatchSize(self.element_count + other.element_count, self.byte_count + other.byte_count)