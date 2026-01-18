class PasswordTruncateError(PasswordSizeError):
    """
    Error raised if password would be truncated by hash.
    This derives from :exc:`PasswordSizeError` (above).

    Hashers such as :class:`~passlib.hash.bcrypt` can be configured to raises
    this error by setting ``truncate_error=True``.

    .. attribute:: max_size

        indicates the maximum allowed size.

    .. versionadded:: 1.7
    """

    def __init__(self, cls, msg=None):
        if msg is None:
            msg = 'Password too long (%s truncates to %d characters)' % (cls.name, cls.truncate_size)
        PasswordSizeError.__init__(self, cls.truncate_size, msg)