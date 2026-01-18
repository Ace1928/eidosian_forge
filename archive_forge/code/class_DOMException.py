from .domreg import getDOMImplementation, registerDOMImplementation
class DOMException(Exception):
    """Abstract base class for DOM exceptions.
    Exceptions with specific codes are specializations of this class."""

    def __init__(self, *args, **kw):
        if self.__class__ is DOMException:
            raise RuntimeError('DOMException should not be instantiated directly')
        Exception.__init__(self, *args, **kw)

    def _get_code(self):
        return self.code