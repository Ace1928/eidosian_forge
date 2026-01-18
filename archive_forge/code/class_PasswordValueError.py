class PasswordValueError(ValueError):
    """
    Error raised if a password can't be hashed / verified for various reasons.
    This exception derives from the builtin :exc:`!ValueError`.

    May be thrown directly when password violates internal invariants of hasher
    (e.g. some don't support NULL characters).  Hashers may also throw more specific subclasses,
    such as :exc:`!PasswordSizeError`.

    .. versionadded:: 1.7.3
    """
    pass