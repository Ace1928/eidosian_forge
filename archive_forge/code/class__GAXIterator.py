import abc
class _GAXIterator(Iterator):
    """A generic class for iterating through Cloud gRPC APIs list responses.

    Any:
        client (google.cloud.client.Client): The API client.
        page_iter (google.gax.PageIterator): A GAX page iterator to be wrapped
            to conform to the :class:`Iterator` interface.
        item_to_value (Callable[Iterator, Any]): Callable to convert an item
            from the protobuf response into a native object. Will
            be called with the iterator and a single item.
        max_results (int): The maximum number of results to fetch.

    .. autoattribute:: pages
    """

    def __init__(self, client, page_iter, item_to_value, max_results=None):
        super(_GAXIterator, self).__init__(client, item_to_value, page_token=page_iter.page_token, max_results=max_results)
        self._gax_page_iter = page_iter

    def _next_page(self):
        """Get the next page in the iterator.

        Wraps the response from the :class:`~google.gax.PageIterator` in a
        :class:`Page` instance and captures some state at each page.

        Returns:
            Optional[Page]: The next page in the iterator or :data:`None` if
                  there are no pages left.
        """
        try:
            items = next(self._gax_page_iter)
            page = Page(self, items, self.item_to_value)
            self.next_page_token = self._gax_page_iter.page_token or None
            return page
        except StopIteration:
            return None