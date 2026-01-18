class PasswordSizeError(PasswordValueError):
    """
    Error raised if a password exceeds the maximum size allowed
    by Passlib (by default, 4096 characters); or if password exceeds
    a hash-specific size limitation.

    This exception derives from :exc:`PasswordValueError` (above).

    Many password hash algorithms take proportionately larger amounts of time and/or
    memory depending on the size of the password provided. This could present
    a potential denial of service (DOS) situation if a maliciously large
    password is provided to an application. Because of this, Passlib enforces
    a maximum size limit, but one which should be *much* larger
    than any legitimate password. :exc:`PasswordSizeError` derives
    from :exc:`!ValueError`.

    .. note::
        Applications wishing to use a different limit should set the
        ``PASSLIB_MAX_PASSWORD_SIZE`` environmental variable before
        Passlib is loaded. The value can be any large positive integer.

    .. attribute:: max_size

        indicates the maximum allowed size.

    .. versionadded:: 1.6
    """
    max_size = None

    def __init__(self, max_size, msg=None):
        self.max_size = max_size
        if msg is None:
            msg = 'password exceeds maximum allowed size'
        PasswordValueError.__init__(self, msg)