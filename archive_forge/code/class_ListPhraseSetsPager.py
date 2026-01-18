from typing import (
from google.cloud.speech_v2.types import cloud_speech
class ListPhraseSetsPager:
    """A pager for iterating through ``list_phrase_sets`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.speech_v2.types.ListPhraseSetsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``phrase_sets`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListPhraseSets`` requests and continue to iterate
    through the ``phrase_sets`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.speech_v2.types.ListPhraseSetsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., cloud_speech.ListPhraseSetsResponse], request: cloud_speech.ListPhraseSetsRequest, response: cloud_speech.ListPhraseSetsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.speech_v2.types.ListPhraseSetsRequest):
                The initial request object.
            response (google.cloud.speech_v2.types.ListPhraseSetsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = cloud_speech.ListPhraseSetsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[cloud_speech.ListPhraseSetsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[cloud_speech.PhraseSet]:
        for page in self.pages:
            yield from page.phrase_sets

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)