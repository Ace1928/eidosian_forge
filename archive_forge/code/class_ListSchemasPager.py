from typing import (
from google.pubsub_v1.types import schema
class ListSchemasPager:
    """A pager for iterating through ``list_schemas`` requests.

    This class thinly wraps an initial
    :class:`google.pubsub_v1.types.ListSchemasResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``schemas`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListSchemas`` requests and continue to iterate
    through the ``schemas`` field on the
    corresponding responses.

    All the usual :class:`google.pubsub_v1.types.ListSchemasResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., schema.ListSchemasResponse], request: schema.ListSchemasRequest, response: schema.ListSchemasResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.pubsub_v1.types.ListSchemasRequest):
                The initial request object.
            response (google.pubsub_v1.types.ListSchemasResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = schema.ListSchemasRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[schema.ListSchemasResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[schema.Schema]:
        for page in self.pages:
            yield from page.schemas

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)