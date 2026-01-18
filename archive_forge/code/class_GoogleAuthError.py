class GoogleAuthError(Exception):
    """Base class for all google.auth errors."""

    def __init__(self, *args, **kwargs):
        super(GoogleAuthError, self).__init__(*args)
        retryable = kwargs.get('retryable', False)
        self._retryable = retryable

    @property
    def retryable(self):
        return self._retryable