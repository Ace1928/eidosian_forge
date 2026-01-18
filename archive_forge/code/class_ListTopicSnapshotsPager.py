from typing import (
from google.pubsub_v1.types import pubsub
class ListTopicSnapshotsPager:
    """A pager for iterating through ``list_topic_snapshots`` requests.

    This class thinly wraps an initial
    :class:`google.pubsub_v1.types.ListTopicSnapshotsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``snapshots`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTopicSnapshots`` requests and continue to iterate
    through the ``snapshots`` field on the
    corresponding responses.

    All the usual :class:`google.pubsub_v1.types.ListTopicSnapshotsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., pubsub.ListTopicSnapshotsResponse], request: pubsub.ListTopicSnapshotsRequest, response: pubsub.ListTopicSnapshotsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.pubsub_v1.types.ListTopicSnapshotsRequest):
                The initial request object.
            response (google.pubsub_v1.types.ListTopicSnapshotsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = pubsub.ListTopicSnapshotsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[pubsub.ListTopicSnapshotsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[str]:
        for page in self.pages:
            yield from page.snapshots

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)