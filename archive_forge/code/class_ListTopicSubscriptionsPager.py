from typing import (
from google.cloud.pubsublite_v1.types import admin
from google.cloud.pubsublite_v1.types import common
class ListTopicSubscriptionsPager:
    """A pager for iterating through ``list_topic_subscriptions`` requests.

    This class thinly wraps an initial
    :class:`google.cloud.pubsublite_v1.types.ListTopicSubscriptionsResponse` object, and
    provides an ``__iter__`` method to iterate through its
    ``subscriptions`` field.

    If there are more pages, the ``__iter__`` method will make additional
    ``ListTopicSubscriptions`` requests and continue to iterate
    through the ``subscriptions`` field on the
    corresponding responses.

    All the usual :class:`google.cloud.pubsublite_v1.types.ListTopicSubscriptionsResponse`
    attributes are available on the pager. If multiple requests are made, only
    the most recent response is retained, and thus used for attribute lookup.
    """

    def __init__(self, method: Callable[..., admin.ListTopicSubscriptionsResponse], request: admin.ListTopicSubscriptionsRequest, response: admin.ListTopicSubscriptionsResponse, *, metadata: Sequence[Tuple[str, str]]=()):
        """Instantiate the pager.

        Args:
            method (Callable): The method that was originally called, and
                which instantiated this pager.
            request (google.cloud.pubsublite_v1.types.ListTopicSubscriptionsRequest):
                The initial request object.
            response (google.cloud.pubsublite_v1.types.ListTopicSubscriptionsResponse):
                The initial response object.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        self._method = method
        self._request = admin.ListTopicSubscriptionsRequest(request)
        self._response = response
        self._metadata = metadata

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    @property
    def pages(self) -> Iterator[admin.ListTopicSubscriptionsResponse]:
        yield self._response
        while self._response.next_page_token:
            self._request.page_token = self._response.next_page_token
            self._response = self._method(self._request, metadata=self._metadata)
            yield self._response

    def __iter__(self) -> Iterator[str]:
        for page in self.pages:
            yield from page.subscriptions

    def __repr__(self) -> str:
        return '{0}<{1!r}>'.format(self.__class__.__name__, self._response)