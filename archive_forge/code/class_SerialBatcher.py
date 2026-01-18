from abc import abstractmethod, ABCMeta
from typing import Generic, List, NamedTuple
import asyncio
from google.cloud.pubsublite.internal.wire.connection import Request, Response
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
class SerialBatcher(Generic[Request, Response]):
    _sizer: RequestSizer[Request]
    _requests: List[WorkItem[Request, Response]]
    _batch_size: BatchSize

    def __init__(self, sizer: RequestSizer[Request]=IgnoredRequestSizer()):
        self._sizer = sizer
        self._requests = []
        self._batch_size = BatchSize(0, 0)

    def add(self, request: Request) -> 'asyncio.Future[Response]':
        """Add a new request to this batcher.

        Args:
          request: The request to send.

        Returns:
          A future that will resolve to the response or a GoogleAPICallError.
        """
        item = WorkItem[Request, Response](request)
        self._requests.append(item)
        self._batch_size += self._sizer.get_size(request)
        return item.response_future

    def size(self) -> BatchSize:
        return self._batch_size

    def flush(self) -> List[WorkItem[Request, Response]]:
        requests = self._requests
        self._requests = []
        self._batch_size = BatchSize(0, 0)
        return requests