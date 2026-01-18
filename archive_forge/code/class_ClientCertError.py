class ClientCertError(GoogleAuthError):
    """Used to indicate that client certificate is missing or invalid."""

    @property
    def retryable(self):
        return False