class MissingBackendError(RuntimeError):
    """Error raised if multi-backend handler has no available backends;
    or if specifically requested backend is not available.

    :exc:`!MissingBackendError` derives
    from :exc:`RuntimeError`, since it usually indicates
    lack of an external library or OS feature.
    This is primarily raised by handlers which depend on
    external libraries (which is currently just
    :class:`~passlib.hash.bcrypt`).
    """