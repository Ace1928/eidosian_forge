from typing import Any
class TripWire:
    """Class raising error if used

    Standard use is to proxy modules that we could not import

    Examples
    --------
    >>> a_module = TripWire('We do not have a_module')
    >>> a_module.do_silly_thing('with silly string') #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TripWireError: We do not have a_module
    """

    def __init__(self, msg: str) -> None:
        self._msg = msg

    def __getattr__(self, attr_name: str) -> Any:
        """Raise informative error accessing attributes"""
        raise TripWireError(self._msg)