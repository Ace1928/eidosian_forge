from abc import abstractmethod, ABCMeta
from typing import Generic, List, NamedTuple
import asyncio
from google.cloud.pubsublite.internal.wire.connection import Request, Response
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
class IgnoredRequestSizer(RequestSizer[Request]):

    def get_size(self, request) -> BatchSize:
        return BatchSize(0, 0)