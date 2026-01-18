from typing import Any, AsyncIterator, Awaitable, Callable, Sequence, Tuple, Optional, Iterator
from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
class ListQuantumReservationGrantsPager:
    """A pager for iterating through ``list_quantum_reservation_grants`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``reservation_grants`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListQuantumReservationGrants`` requests and continue to iterate
    through the ``reservation_grants`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., engine.ListQuantumReservationGrantsResponse], request: engine.ListQuantumReservationGrantsRequest, response: engine.ListQuantumReservationGrantsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest):
                The initial request object.
            response (google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = engine.ListQuantumReservationGrantsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[engine.ListQuantumReservationGrantsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[quantum.QuantumReservationGrant]:
        for page in self.pages:
            yield from page.reservation_grants

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self._response!r}>'