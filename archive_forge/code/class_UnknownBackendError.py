class UnknownBackendError(ValueError):
    """
    Error raised if multi-backend handler doesn't recognize backend name.
    Inherits from :exc:`ValueError`.

    .. versionadded:: 1.7
    """

    def __init__(self, hasher, backend):
        self.hasher = hasher
        self.backend = backend
        message = '%s: unknown backend: %r' % (hasher.name, backend)
        ValueError.__init__(self, message)