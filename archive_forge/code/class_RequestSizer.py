from abc import abstractmethod, ABCMeta
from typing import Generic, List, NamedTuple
import asyncio
from google.cloud.pubsublite.internal.wire.connection import Request, Response
from google.cloud.pubsublite.internal.wire.work_item import WorkItem
class RequestSizer(Generic[Request], metaclass=ABCMeta):
    """A RequestSizer determines the size of a request."""

    @abstractmethod
    def get_size(self, request: Request) -> BatchSize:
        """
        Args:
          request: A single request.

        Returns: The BatchSize of this request
        """
        raise NotImplementedError()