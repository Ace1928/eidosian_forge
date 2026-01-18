from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types import spanner
class ListSessionsAsyncPager:
    """A pager for iterating through ``list_sessions`` requests.

    This class thinly wraps an initial
    :class:`googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.ListSessionsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``sessions`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListSessions`` requests and continue to iterate
    through the ``sessions`` field on the
    corresponding responses.

    All the usual :class:`googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.ListSessionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[spanner.ListSessionsResponse]], request: spanner.ListSessionsRequest, response: spanner.ListSessionsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.ListSessionsRequest):
                The initial request object.
            response (googlecloudsdk.generated_clients.gapic_clients.spanner_v1.types.ListSessionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = spanner.ListSessionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[spanner.ListSessionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[spanner.Session]:

        async def async_generator():
            async for page in self.pages:
                for response in page.sessions:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)