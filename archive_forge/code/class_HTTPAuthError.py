class HTTPAuthError(HTTPError):
    """Raised for 401 Unauthorized responses from the server."""

    def __init__(self, message, status_code=401):
        super(HTTPError, self).__init__(message, status_code)