import abc
def _get_next_page_response(self):
    """Requests the next page from the path provided.

        Returns:
            dict: The parsed JSON response of the next page's contents.

        Raises:
            ValueError: If the HTTP method is not ``GET`` or ``POST``.
        """
    params = self._get_query_params()
    if self._HTTP_METHOD == 'GET':
        return self.api_request(method=self._HTTP_METHOD, path=self.path, query_params=params)
    elif self._HTTP_METHOD == 'POST':
        return self.api_request(method=self._HTTP_METHOD, path=self.path, data=params)
    else:
        raise ValueError('Unexpected HTTP method', self._HTTP_METHOD)