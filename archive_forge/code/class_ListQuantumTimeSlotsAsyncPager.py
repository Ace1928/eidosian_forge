from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
class ListQuantumTimeSlotsAsyncPager:
    """A pager for iterating through ``list_quantum_time_slots`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``time_slots`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumTimeSlots`` requests and continue to iterate
    through the ``time_slots`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[engine.ListQuantumTimeSlotsResponse]], request: engine.ListQuantumTimeSlotsRequest, response: engine.ListQuantumTimeSlotsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest):
                The initial request object.
            response (google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = engine.ListQuantumTimeSlotsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumTimeSlotsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumTimeSlot]:

        async def async_generator():
            async for page in self.pages:
                for response in page.time_slots:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._response!r}>'