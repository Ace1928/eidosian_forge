import sys
class SAXException(Exception):
    """Encapsulate an XML error or warning. This class can contain
    basic error or warning information from either the XML parser or
    the application: you can subclass it to provide additional
    functionality, or to add localization. Note that although you will
    receive a SAXException as the argument to the handlers in the
    ErrorHandler interface, you are not actually required to raise
    the exception; instead, you can simply read the information in
    it."""

    def __init__(self, msg, exception=None):
        """Creates an exception. The message is required, but the exception
        is optional."""
        self._msg = msg
        self._exception = exception
        Exception.__init__(self, msg)

    def getMessage(self):
        """Return a message for this exception."""
        return self._msg

    def getException(self):
        """Return the embedded exception, or None if there was none."""
        return self._exception

    def __str__(self):
        """Create a string representation of the exception."""
        return self._msg

    def __getitem__(self, ix):
        """Avoids weird error messages if someone does exception[ix] by
        mistake, since Exception has __getitem__ defined."""
        raise AttributeError('__getitem__')