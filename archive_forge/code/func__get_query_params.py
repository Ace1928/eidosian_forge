import abc
def _get_query_params(self):
    """Getter for query parameters for the next request.

        Returns:
            dict: A dictionary of query parameters.
        """
    result = {}
    if self.next_page_token is not None:
        result[self._PAGE_TOKEN] = self.next_page_token
    page_size = None
    if self.max_results is not None:
        page_size = self.max_results - self.num_results
        if self._page_size is not None:
            page_size = min(page_size, self._page_size)
    elif self._page_size is not None:
        page_size = self._page_size
    if page_size is not None:
        result[self._MAX_RESULTS] = page_size
    result.update(self.extra_params)
    return result