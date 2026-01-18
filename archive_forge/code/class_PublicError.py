class PublicError(RuntimeError):
    """An error whose text does not contain sensitive information.

    Fields:
      http_code: Integer between 400 and 599 inclusive (e.g., 404).
      headers: List of additional key-value pairs to include in the
        response body, like `[("Allow", "GET")]` for HTTP 405 or
        `[("WWW-Authenticate", "Digest")]` for HTTP 401. May be empty.
    """
    http_code = 500

    def __init__(self, details):
        super().__init__(details)
        self.headers = []