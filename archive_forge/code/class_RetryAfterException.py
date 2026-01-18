class RetryAfterException(ClientException):
    """The base exception for ClientExceptions that use Retry-After header."""

    def __init__(self, *args, **kwargs):
        try:
            self.retry_after = int(kwargs.pop('retry_after'))
        except (KeyError, ValueError):
            self.retry_after = 0
        super(RetryAfterException, self).__init__(*args, **kwargs)