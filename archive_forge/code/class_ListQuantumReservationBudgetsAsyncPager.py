from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
class ListQuantumReservationBudgetsAsyncPager:
    """A pager for iterating through ``list_quantum_reservation_budgets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse` object, and
    provides an ``__aiter__`` method to iterate through its
    ``reservation_budgets`` field.

    If there are more pages, the ``__aiter__`` method will make additional
    ``ListQuantumReservationBudgets`` requests and continue to iterate
    through the ``reservation_budgets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., Awaitable[engine.ListQuantumReservationBudgetsResponse]], request: engine.ListQuantumReservationBudgetsRequest, response: engine.ListQuantumReservationBudgetsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiates the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest):
                The initial request object.
            response (google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = engine.ListQuantumReservationBudgetsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    async def pages(self) -> AsyncIterator[engine.ListQuantumReservationBudgetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = await self._method(self._request, metadata=self._metadata)
            yield self._response

    def __aiter__(self) -> AsyncIterator[quantum.QuantumReservationBudget]:

        async def async_generator():
            async for page in self.pages:
                for response in page.reservation_budgets:
                    yield response
        return async_generator()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._response!r}>'