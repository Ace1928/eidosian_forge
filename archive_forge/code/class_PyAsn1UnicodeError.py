class PyAsn1UnicodeError(PyAsn1Error, UnicodeError):
    """Unicode text processing error

    The `PyAsn1UnicodeError` exception is a base class for errors relating to
    unicode text de/serialization.

    Apart from inheriting from :class:`PyAsn1Error`, it also inherits from
    :class:`UnicodeError` to help the caller catching unicode-related errors.
    """

    def __init__(self, message, unicode_error=None):
        if isinstance(unicode_error, UnicodeError):
            UnicodeError.__init__(self, *unicode_error.args)
        PyAsn1Error.__init__(self, message)